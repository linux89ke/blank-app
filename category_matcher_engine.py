import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import logging
import traceback
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def compile_rules_from_json(raw_rules: list, code_to_path: dict = None) -> dict:
    if code_to_path is None:
        code_to_path = {}

    compiled = {}
    for rule in raw_rules:
        positive_kws = rule.get('positive', {})
        if not positive_kws:
            continue

        code_str = str(rule.get('category_code', ''))
        cat_key = code_to_path.get(code_str, rule.get('category_name', '')).lower().strip()
        if not cat_key:
            continue

        pattern = re.compile(
            r'\b(' + '|'.join(re.escape(k.lower()) for k in positive_kws) + r')\b'
        )

        compiled[cat_key] = {
            'pattern': pattern,
            'weights': {k.lower(): float(v) for k, v in positive_kws.items()}
        }

    return compiled


class CategoryMatcherEngine:
    def __init__(self, db_path="cat_learning.db"):
        self.db_path = db_path
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            stop_words='english'
        )
        self.tfidf_matrix = None
        self.categories = []
        self._tfidf_built = False
        self.learning_db = {}
        self.compiled_rules = {}  
        self.correction_classifier = None
        self.correction_vectorizer = None
        self._init_db()
        self.load_learning_db()

    def predict_batch(self, names: list) -> dict:
        if not self._tfidf_built or not names:
            return {}

        # Ensure path lookup exists for bypass logic
        if not hasattr(self, '_path_lookup') or not self._path_lookup:
            self._path_lookup = {}
            for cat in self.categories:
                cl = cat.lower()
                self._path_lookup[cl] = cat
                leaf = cl
                for sep in ('/', '>'):
                    if sep in leaf:
                        leaf = leaf.split(sep)[-1]
                self._path_lookup[leaf.strip()] = cat

        processed = [clean_text(n) for n in names]
        X = self.vectorizer.transform(processed)
        similarities_matrix = cosine_similarity(X, self.tfidf_matrix) 
        
        results = {}
        for i, name in enumerate(names):
            learned = self.predict_category_from_learning(name)
            if learned:
                results[name] = (learned, 1.0)
                continue
                
            similarities = similarities_matrix[i]
            top_indices = similarities.argsort()[-20:][::-1]
            
            best_category = ""
            best_score = -1.0
            name_lower = str(name).lower()
            
            # 1. Evaluate Top 20 TF-IDF matches
            for idx in top_indices:
                cat_path = self.categories[idx]
                base_score = float(similarities[idx])
                boost = 0.0
                
                cat_path_lower = cat_path.lower()
                leaf_lower = cat_path_lower
                for sep in ('/', '>'):
                    if sep in leaf_lower:
                        leaf_lower = leaf_lower.split(sep)[-1]
                leaf_lower = leaf_lower.strip()
                
                rule = self.compiled_rules.get(cat_path_lower) or self.compiled_rules.get(leaf_lower)
                if rule:
                    matches = rule['pattern'].findall(name_lower)
                    if matches:
                        boost = sum(rule['weights'].get(m.lower(), 0.0) for m in set(matches))
                        
                final_score = base_score + (boost * 0.6)
                if final_score > best_score:
                    best_score = final_score
                    best_category = cat_path

            # 2. Evaluate ALL JSON rules to completely bypass TF-IDF blindspots
            if self.compiled_rules:
                for rule_key, rule in self.compiled_rules.items():
                    matches = rule['pattern'].findall(name_lower)
                    if matches:
                        boost = sum(rule['weights'].get(m.lower(), 0.0) for m in set(matches))
                        if boost > 0:
                            final_score = (boost * 0.6) # Bypass score
                            if final_score > best_score:
                                mapped_cat = self._path_lookup.get(rule_key)
                                if mapped_cat:
                                    best_score = final_score
                                    best_category = mapped_cat
                    
            _threshold = 0.35 if getattr(self, '_index_has_full_paths', True) else 0.15
            if best_score < _threshold:
                results[name] = ("", 0.0)
            else:
                results[name] = (best_category, best_score)
                
        return results

    def set_compiled_rules(self, rules, code_to_path: dict = None):
        if isinstance(rules, list):
            self.compiled_rules = compile_rules_from_json(rules, code_to_path or {})
        else:
            self.compiled_rules = rules or {}

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('''
                    CREATE TABLE IF NOT EXISTS category_corrections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        category TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to init category learning DB: {e}")

    def load_learning_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT name, category FROM category_corrections", conn)
                if not df.empty:
                    self.learning_db = df.groupby('name')['category'].last().to_dict()
                    self._retrain_correction_classifier(df)
        except Exception as e:
            logger.warning(f"Failed to load category learning DB: {e}")

    def _retrain_correction_classifier(self, df=None):
        if df is None:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    df = pd.read_sql_query("SELECT name, category FROM category_corrections", conn)
            except Exception:
                return
        if df is None or df.empty or len(df['category'].unique()) < 2:
            return
        try:
            df['clean_name'] = df['name'].apply(clean_text)
            self.correction_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
            X = self.correction_vectorizer.fit_transform(df['clean_name'])
            y = df['category']
            self.correction_classifier = LogisticRegression(class_weight='balanced', max_iter=1000)
            self.correction_classifier.fit(X, y)
        except Exception as e:
            logger.warning(f"Failed to retrain correction classifier: {e}")

    def apply_learned_correction(self, name: str, category: str, auto_save=True):
        clean_n = clean_text(name)
        if not clean_n or not category: return
        self.learning_db[clean_n] = category
        if auto_save:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    c = conn.cursor()
                    c.execute("INSERT INTO category_corrections (name, category) VALUES (?, ?)", (clean_n, category))
                    conn.commit()
                self._retrain_correction_classifier()
            except Exception as e:
                logger.warning(f"Failed to save correction to DB: {e}")

    def save_learning_db(self):
        if not self.learning_db: return
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute("BEGIN TRANSACTION")
                for name, cat in self.learning_db.items():
                    c.execute("INSERT INTO category_corrections (name, category) VALUES (?, ?)", (name, cat))
                conn.commit()
            self._retrain_correction_classifier()
        except Exception as e:
            logger.warning(f"Failed to batch save learning DB: {e}")

    def build_tfidf_index(self, categories_list: list):
        if not categories_list: return
        self.categories = [str(c).strip() for c in categories_list if str(c).strip() and str(c).strip().lower() != 'nan']
        if not self.categories: return
        clean_cats = [clean_text(c) for c in self.categories]
        sep_count = sum(1 for c in self.categories if '/' in c or '>' in c)
        self._index_has_full_paths = (sep_count / max(len(self.categories), 1)) > 0.3
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(clean_cats)
            self._tfidf_built = True
            logger.info(f'[TF-IDF] Built index: {len(self.categories)} categories, '
                        f'full_paths={self._index_has_full_paths} '
                        f'(sep_count={sep_count})')
        except Exception as e:
            logger.warning(f"Failed to build TF-IDF index: {e}")

    def predict_category_from_learning(self, name: str) -> str:
        clean_n = clean_text(name)
        if clean_n in self.learning_db:
            return self.learning_db[clean_n]
        if self.correction_classifier and self.correction_vectorizer:
            try:
                vec = self.correction_vectorizer.transform([clean_n])
                probs = self.correction_classifier.predict_proba(vec)[0]
                max_prob_idx = np.argmax(probs)
                if probs[max_prob_idx] > 0.6: 
                    return self.correction_classifier.classes_[max_prob_idx]
            except Exception:
                pass
        return None

    def get_category_with_fallback(self, name: str, kw_map: dict = None, categories_list: list = None) -> str:
        learned = self.predict_category_from_learning(name)
        if learned: return learned
        
        if self._tfidf_built:
            try:
                name_clean = clean_text(name)
                name_vec = self.vectorizer.transform([name_clean])
                similarities = cosine_similarity(name_vec, self.tfidf_matrix).flatten()
                best_idx = np.argmax(similarities)
                if similarities[best_idx] > 0.35:
                    return self.categories[best_idx]
            except Exception:
                pass
                
        if kw_map:
            name_lower = str(name).lower()
            for kw, cat in kw_map.items():
                if re.search(r'\b' + re.escape(kw) + r'\b', name_lower):
                    return cat
        return ""

    def get_category_with_boost(self, name: str, top_n: int = 20) -> str:
        learned = self.predict_category_from_learning(name)
        if learned: return learned

        if not getattr(self, '_tfidf_built', False):
            return ""

        if not hasattr(self, '_path_lookup') or not self._path_lookup:
            self._path_lookup = {}
            for cat in self.categories:
                cl = cat.lower()
                self._path_lookup[cl] = cat
                leaf = cl
                for sep in ('/', '>'):
                    if sep in leaf:
                        leaf = leaf.split(sep)[-1]
                self._path_lookup[leaf.strip()] = cat
        
        try:
            name_clean = clean_text(name)
            name_vec = self.vectorizer.transform([name_clean])
            similarities = cosine_similarity(name_vec, self.tfidf_matrix).flatten()
            
            top_indices = similarities.argsort()[-top_n:][::-1]
            
            best_category = ""
            best_score = -1.0
            name_lower = str(name).lower()
            
            for idx in top_indices:
                cat_path = self.categories[idx]
                base_score = float(similarities[idx])
                boost = 0.0
                
                cat_path_lower = cat_path.lower()
                leaf_lower = cat_path_lower
                for sep in ('/', '>'):
                    if sep in leaf_lower:
                        leaf_lower = leaf_lower.split(sep)[-1]
                leaf_lower = leaf_lower.strip()
                
                rule = self.compiled_rules.get(cat_path_lower) or self.compiled_rules.get(leaf_lower)
                if rule:
                    matches = rule['pattern'].findall(name_lower)
                    if matches:
                        boost = sum(rule['weights'].get(m.lower(), 0.0) for m in set(matches))
                
                final_score = base_score + (boost * 0.6) 
                if final_score > best_score:
                    best_score = final_score
                    best_category = cat_path

            if self.compiled_rules:
                for rule_key, rule in self.compiled_rules.items():
                    matches = rule['pattern'].findall(name_lower)
                    if matches:
                        boost = sum(rule['weights'].get(m.lower(), 0.0) for m in set(matches))
                        if boost > 0:
                            final_score = (boost * 0.6)
                            if final_score > best_score:
                                mapped_cat = self._path_lookup.get(rule_key)
                                if mapped_cat:
                                    best_score = final_score
                                    best_category = mapped_cat
            
            _threshold = 0.35 if getattr(self, '_index_has_full_paths', True) else 0.15
            if best_score < _threshold:
                return ""
                
            return best_category
            
        except Exception as e:
            logger.warning(f"Boosted prediction failed: {e}")
            return ""

    def build_keyword_to_category_mapping(self) -> dict:
        kw_map = {}
        for cat in self.categories:
            parts = [p.strip().lower() for p in cat.split('>')]
            if len(parts) > 1:
                kw_map[parts[-1]] = cat
        return kw_map


_engine_instance = None

def get_engine(db_path="cat_learning.db"):
    global _engine_instance
    if _engine_instance is None:
        try:
            _engine_instance = CategoryMatcherEngine(db_path)
        except Exception as e:
            logger.error(f"Failed to initialize CategoryMatcherEngine: {e}")
            logger.error(traceback.format_exc())
            _engine_instance = None
    return _engine_instance

def check_wrong_category(data: pd.DataFrame, categories_list: list, compiled_rules: dict = None, cat_path_to_code: dict = None, code_to_path: dict = None, confidence_threshold: float = 0.0):
    if not {'NAME', 'CATEGORY'}.issubset(data.columns) or not categories_list:
        return pd.DataFrame(columns=data.columns)
        
    engine = get_engine()
    if engine is None:
        return pd.DataFrame(columns=data.columns)

    _effective_cats = categories_list
    if code_to_path:
        full_paths = list(code_to_path.values())
        sep_ratio = sum(1 for p in full_paths if '/' in p or '>' in p) / max(len(full_paths), 1)
        if sep_ratio > 0.3:
            _effective_cats = full_paths
            logger.info(f'[WrongCat] Using code_to_path full paths '
                        f'({len(_effective_cats)}) instead of categories_list '
                        f'({len(categories_list)}) for TF-IDF index')

    if not engine._tfidf_built:
        engine.build_tfidf_index(_effective_cats)
    elif not getattr(engine, '_index_has_full_paths', False) and _effective_cats is not categories_list:
        logger.info('[WrongCat] Rebuilding TF-IDF index with full paths')
        engine._tfidf_built = False
        engine.build_tfidf_index(_effective_cats)

    if not getattr(engine, '_index_has_full_paths', False):
        logger.warning(
            '[WrongCat] Skipping wrong-category detection: TF-IDF index was built '
            'on bare leaf category names (no category_map.xlsx full paths available). '
            'Ensure category_map.xlsx is present and loading correctly.'
        )
        return pd.DataFrame(columns=data.columns)

    if compiled_rules:
        engine.set_compiled_rules(compiled_rules)

    if cat_path_to_code is None: cat_path_to_code = {}
    if code_to_path is None: code_to_path = {}

    d = data.copy()
    d['_cat_clean'] = d['CATEGORY'].astype(str).str.strip()
    
    if 'CATEGORY_CODE' in d.columns and code_to_path:
        for idx, row in d.iterrows():
            if not row['_cat_clean'] or row['_cat_clean'].lower() in ('nan', 'none', 'miscellaneous'):
                code = str(row.get('CATEGORY_CODE', '')).strip().split('.')[0]
                if code in code_to_path:
                    d.at[idx, '_cat_clean'] = code_to_path[code]

    d['_cat_lower'] = d['_cat_clean'].str.lower()
    d['_name_clean'] = d['NAME'].astype(str).str.strip()
    
    leaf_to_full_path = {}
    if code_to_path:
        for full_path in code_to_path.values():
            for sep in ('/', '>'):
                if sep in full_path:
                    leaf = full_path.split(sep)[-1].strip().lower()
                    break
            else:
                leaf = full_path.strip().lower()
            if leaf not in leaf_to_full_path:
                leaf_to_full_path[leaf] = full_path

    flagged_indices = []
    comment_map = {}
    kw_map = engine.build_keyword_to_category_mapping()

    # Get predictions in BULK
    unique_names = d['_name_clean'].unique().tolist()
    if hasattr(engine, 'predict_batch') and callable(engine.predict_batch):
        batch_preds = engine.predict_batch(unique_names)
    else:
        batch_preds = {}

    for idx, row in d.iterrows():
        current_cat = row['_cat_clean']
        name = row['_name_clean']
        
        if not current_cat or current_cat.lower() in ('nan', 'none', ''):
            continue

        if 'miscellaneous' in current_cat.lower():
            flagged_indices.append(idx)
            comment_map[idx] = "Category is 'Miscellaneous'"
            continue

        predicted = ""
        if batch_preds and name in batch_preds:
            predicted, _score = batch_preds[name]
        else:
            predicted = engine.get_category_with_boost(name)
            if not predicted:
                predicted = engine.get_category_with_fallback(name, kw_map, categories_list)

        if predicted and code_to_path and '/' not in predicted and '>' not in predicted:
            predicted = leaf_to_full_path.get(predicted.strip().lower(), predicted)
        
        if predicted and predicted.lower() != current_cat.lower():
            def get_top(path):
                for sep in ('/', '>'):
                    if sep in path:
                        return path.split(sep)[0].strip().lower()
                return path.strip().lower()

            def get_leaf(path):
                for sep in ('/', '>'):
                    if sep in path:
                        return path.split(sep)[-1].strip().lower()
                return path.strip().lower()

            p_leaf = get_leaf(predicted)
            c_leaf = get_leaf(current_cat)

            if c_leaf in predicted.lower():
                continue

            current_full = current_cat
            if code_to_path:
                row_code = str(row.get('CATEGORY_CODE', '')).strip().split('.')[0]
                if row_code and row_code in code_to_path:
                    current_full = code_to_path[row_code]
                else:
                    code = cat_path_to_code.get(current_cat.lower(), '')
                    if code and code in code_to_path:
                        current_full = code_to_path[code]
                    else:
                        resolved = leaf_to_full_path.get(current_cat.strip().lower())
                        if resolved:
                            current_full = resolved

            def get_segments(path, n):
                for sep in ('/', '>'):
                    if sep in path:
                        parts = [p.strip().lower() for p in path.split(sep)]
                        return tuple(parts[:n])
                return (path.strip().lower(),)

            p_segs = get_segments(predicted, 3)
            c_segs = get_segments(current_full, 3)
            shared = sum(1 for a, b in zip(p_segs, c_segs) if a == b)
            if shared >= min(2, len(p_segs), len(c_segs)):
                continue

            c_leaf_lower = current_cat.strip().lower()
            c_full_lower = current_full.strip().lower()
            p_top_lower = get_top(predicted).strip().lower()

            _SAME_DOMAIN_CATEGORIES = {
                'health & beauty': {
                    'creams', 'strips', 'supplements', 'creams & moisturizers',
                    'conditioners', 'face moisturizers', 'cleansers', 'soaps & cleansers',
                    'hair & scalp treatments', 'toners', 'face', 'body',
                    'cellulite massagers', 'serums', 'shaving creams', 'gels',
                    'wrinkle & anti-aging devices', 'lips', 'soaps', 'washes',
                    'body wash', 'joint & muscle pain relief', 'bubble bath', 'lotions',
                    'essential oils', 'health & fitness', 'detox & cleanse', 'oils',
                    'sets & kits', 'shaving gels', 'hair sprays', 'eau de parfum',
                    'skin care', 'salon & spa chairs', 'massage chairs', 'heating pads',
                    'makeup sets', 'foundation', 'face primer', 'makeup organizers',
                    'hair color', 'back braces', 'cellulite massagers', 'serums',
                    'face primer', 'hairpieces', 'wrinkle & anti-aging devices', 'bubble bath',
                    'body wash', 'shaving creams', 'soaps', 'washes', 'body',
                    'face', 'lips', 'gels', 'essential oils', 'detox & cleanse',
                    'health & fitness', 'body scrubs', 'nail care', 'eye care',
                    'feminine care', 'oral care', 'medical supplies',
                },
                'home & office': {
                    'bestselling books', 'faith & spirituality',
                    'sets & kits',
                    'medical support hose',
                    'push & pull toys', 'stacking & nesting toys',
                    "women's",
                    'freezers', 'food processors', 'mixers & blenders', 'rice cookers',
                    'deep fryers', 'air fryers', 'cookers', 'microwave ovens',
                    'electric pressure cookers', 'pressure cookers', 'hot pots',
                    'waffle makers', 'toasters', 'kettles', 'coffee makers',
                    'vacuum cleaners', 'wet & dry vacuums', 'bagless vacuum cleaner',
                    'washing machines', 'dishwashers',
                    'standing shelf units', 'coat racks',
                    'printer cutters', 'art set', 'canvas boards & panels',
                    'kitchen utensils & gadgets', 'kitchen storage & organization accessories',
                    'stemmed water glasses', 'whisks', 'wastebasket bags',
                    'bedding sets', 'curtain panels', 'duvet covers', 'mosquito net',
                    'usb fans',
                    'sprayers', 'security & filtering',
                },
                'electronics': {
                    'bluetooth speakers', 'bluetooth headsets', 'earphones & headsets',
                    'portable bluetooth speakers', 'sound bars', 'headphone amplifiers',
                    'earbud headphones', 'headphone extension cables',
                    'wireless lavalier microphones',
                    'smart tvs', 'overhead projectors',
                    'ceiling fans', 'ceiling fan light kits', 'usb fans',
                    'tv remote controls', 'remote controls',
                    'gadgets',
                },
                'phones & tablets': {
                    'chargers', 'earbud headphones', 'rubber strap',
                    'electrical device mounts', 'earphones & headsets',
                    'cell phones', 'android phones', 'smartphones',
                    'flip cases', 'cases', 'screen protectors',
                },
                'fashion': {
                    'sandals', 'sneakers', 'slippers', 'shoes', 'rain boots', 'boots',
                    'casual dresses', 'hats & caps', 'briefs', 'thongs', 'socks',
                    'unisex fabrics', 'stockings', 'polos', 'bras', 'underwear',
                    't-shirts', 'shirts', 'outerwear', 'clothing', 'dresses',
                    'jackets', 'coats', 'jeans',
                    'handbags', 'jewellery',
                },
                'computing': {
                    'laptops', 'desktops', 'tablets', 'monitors', 'keyboards',
                    'mice', 'printers', 'scanners', 'hard drives', 'ssds',
                    'computer accessories', 'networking', 'routers',
                    'portable power banks', 'bluetooth headsets',
                },
                'musical instruments': {
                    'subwoofers', 'bags, cases & covers', 'racks & stands', 'musicals',
                    'microphones', 'amplifiers', 'mixers',
                },
                'grocery': {
                    'standard batteries',
                },
                'baby products': {
                    'pillows', 'lumbar supports', 'wipes, napkins & serviettes',
                    'walkers', 'feminine washes', 'baby formula', 'diapers',
                    'baby monitors', 'strollers',
                },
                'gaming': {
                    'gaming headsets', 'gaming mice', 'gaming keyboards',
                    'controllers', 'ps 5 games', 'ps4 games', 'xbox games',
                    'pc gaming', 'gaming chairs', 'gaming desks',
                },
            }
            same_domain_cats = _SAME_DOMAIN_CATEGORIES.get(p_top_lower, set())
            if c_leaf_lower in same_domain_cats:
                continue

            _CROSS_DOMAIN_BLOCKS = [
                ({'supplements', 'tablets', 'capsules', 'vitamins', 'syrup', 'herbal',
                  'herbs', 'strips', 'milk substitutes'},
                 {'phones & tablets', 'electronics', 'automobile',
                  'industrial & scientific', 'sporting goods'}),

                ({'fashion', 'clothing', 'outerwear', 'apparel', 'shoes', 'footwear',
                  'sneakers', 'slippers', 'socks', 'polos', 'bras', 'underwear',
                  't-shirts', 'shirts', 'dresses', 'jackets', 'coats', 'jeans',
                  'sandals', 'rain boots', 'boots', 'stockings'},
                 {'grocery', 'industrial & scientific', 'automobile',
                  'sporting goods', 'electronics', 'home & office', 'pet supplies'}),

                ({'electronics', 'cell phones', 'bluetooth speakers', 'bluetooth headsets',
                  'earphones', 'headsets', 'smart watches', 'wrist watches', 'tv remote',
                  'remote controls', 'wi-fi', 'dongles', 'power banks', 'earbuds',
                  'headphones', 'laptops', 'cameras', 'speakers', 'portable bluetooth'},
                 {'grocery', 'automobile', 'industrial & scientific',
                  'garden & outdoors', 'sporting goods', 'fashion', 'pet supplies'}),

                ({'wrist watches', "women's watches", "men's watches", 'kids watches',
                  'smart watches', 'wall clocks', 'alarm clocks'},
                 {'fashion', 'sporting goods', 'grocery', 'automobile'}),

                ({'health', 'beauty', 'skin care', 'creams', 'makeup', 'foundation',
                  'heating pads', 'salon & spa', 'salon', 'spa', 'massage', 'medical',
                  'shaving gels', 'hair sprays', 'eau de parfum', 'fragrance', 'perfume',
                  'sets & kits'},
                 {'grocery', 'industrial & scientific', 'sporting goods',
                  'automobile', 'phones & tablets', 'toys & games', 'pet supplies'}),

                ({'home', 'kitchen', 'storage', 'cleaning', 'toilet', 'coat racks',
                  'sewing machines', 'pressure cookers', 'electric pressure cookers',
                  'cookers', 'christian books', 'books', 'printer cutters', 'sprayers',
                  'art set', 'security & filtering'},
                 {'grocery', 'sporting goods', 'automobile',
                  'industrial & scientific', 'garden & outdoors'}),

                ({'outdoor safety', 'play yard', 'baby', 'strollers', 'nursery'},
                 {'garden & outdoors', 'sporting goods', 'automobile'}),

                ({'salon & spa chairs', 'massage chairs'},
                 {'health & beauty'}),
                ({'cell phones', 'earphones & headsets'},
                 {'phones & tablets'}),
                ({'pressure cookers', 'electric pressure cookers'},
                 {'home & office'}),

                ({'creams', 'strips', 'supplements', 'creams & moisturizers'},
                 {'sporting goods', 'automobile', 'grocery',
                  'phones & tablets', 'industrial & scientific'}),

                ({'bluetooth headsets', 'tv remote controls', 'remote controls',
                  'android phones', 'musicals'},
                 {'sporting goods', 'grocery', 'automobile', 'garden & outdoors',
                  'industrial & scientific', 'fashion', 'pet supplies'}),

                ({'christian books & bibles', 'motivational & self-help',
                  'business & economics'},
                 {'home & office', 'industrial & scientific', 'automobile',
                  'grocery', 'sporting goods'}),

                ({'freezers', 'mixers & blenders', 'food processors', 'rice cookers',
                  'bakeware sets', 'utensils', 'printer cutters', 'art set',
                  'push & pull toys'},
                 {'automobile', 'sporting goods', 'grocery',
                  'industrial & scientific', 'garden & outdoors'}),

                ({'stick umbrellas', 'umbrellas'},
                 {'fashion', 'grocery', 'automobile', 'sporting goods'}),

                ({'backpacks', 'camping backpacks', 'bags'},
                 {'electronics', 'automobile', 'industrial & scientific'}),

                ({'digital games', 'ps 5 games', 'ps4 games', 'xbox games'},
                 {'health & beauty', 'grocery', 'automobile',
                  'industrial & scientific', 'fashion'}),

                ({'dyes', 'hair dye', 'fabric dye'},
                 {'toys & games', 'grocery', 'automobile', 'industrial & scientific'}),
            ]
            blocked = False
            for current_kws, forbidden_tops in _CROSS_DOMAIN_BLOCKS:
                if any(kw in c_leaf_lower or kw in c_full_lower for kw in current_kws):
                    if any(p_top_lower.startswith(ft) for ft in forbidden_tops):
                        blocked = True
                        break
            if blocked:
                continue

            if p_leaf != c_leaf:
                flagged_indices.append(idx)
                comment_map[idx] = f"Wrong Category. Suggested: {predicted}"

    if not flagged_indices:
        return pd.DataFrame(columns=data.columns)

    res = data.loc[flagged_indices].copy()
    res['Comment_Detail'] = res.index.map(comment_map)
    return res.drop_duplicates(subset=['PRODUCT_SET_SID'])

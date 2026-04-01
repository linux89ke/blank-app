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
    """
    Converts raw JSON category rules into the compiled format the engine expects.

    Each rule in raw_rules should look like:
        {
            "category_name": "Car Polishes & Waxes",
            "category_code": 1000089,
            "positive": {"car": 2, "polish": 3, "wax": 3}
        }

    Args:
        raw_rules:    List of rule dicts loaded from your JSON file.
        code_to_path: Optional dict mapping str(category_code) -> full category path
                      e.g. {"1000089": "Automobile > Car Care > Car Polishes & Waxes"}
                      If provided, the full path is used as the lookup key so it
                      aligns with what the TF-IDF index stores in engine.categories.

    Returns:
        A dict keyed by lowercase category path (or name), ready for set_compiled_rules().
    """
    if code_to_path is None:
        code_to_path = {}

    compiled = {}
    for rule in raw_rules:
        positive_kws = rule.get('positive', {})
        if not positive_kws:
            continue

        # Resolve the key: prefer the full path from code_to_path, fall back to category_name
        code_str = str(rule.get('category_code', ''))
        cat_key = code_to_path.get(code_str, rule.get('category_name', '')).lower().strip()
        if not cat_key:
            continue

        # Build one regex pattern covering all positive keywords
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
        self.compiled_rules = {}  # Store JSON rules directly in the engine
        self.correction_classifier = None
        self.correction_vectorizer = None
        self._init_db()
        self.load_learning_db()

    def set_compiled_rules(self, rules, code_to_path: dict = None):
        """
        Loads heuristic rules into the engine.

        Accepts either:
          - A raw list of JSON rule dicts (auto-compiled via compile_rules_from_json)
          - An already-compiled dict (from a prior compile_rules_from_json call)
        """
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
        # Detect if index is built on full paths (contains separators) or bare leaves.
        # Full paths give much richer TF-IDF signal; bare leaves score near-zero.
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
        """
        Gets the top N TF-IDF predictions, applies internal JSON heuristic boosts, 
        and returns the highest scoring category.
        """
        learned = self.predict_category_from_learning(name)
        if learned: 
            return learned

        if not getattr(self, '_tfidf_built', False):
            return ""
        
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
                
                # 1. Convert the engine's category path to lowercase
                cat_path_lower = cat_path.lower()
                
                # 2. Check the JSON rules — try full path first, then leaf name as fallback.
                #    This handles both: rules keyed by full path ("automobile > car care > car polishes & waxes")
                #    and rules keyed by bare category_name ("car polishes & waxes").
                leaf_lower = cat_path_lower.split('>')[-1].strip()
                rule = self.compiled_rules.get(cat_path_lower) or self.compiled_rules.get(leaf_lower)
                if rule:
                    matches = rule['pattern'].findall(name_lower)
                    if matches:
                        boost = sum(rule['weights'].get(m.lower(), 0.0) for m in set(matches))
                
                final_score = base_score + (boost * 0.6) # Increased multiplier slightly for more authority
                
                if final_score > best_score:
                    best_score = final_score
                    best_category = cat_path
            
            # Reject garbage matches if confidence is too low.
            # Bare-leaf indexes score much lower than full-path indexes,
            # so use a lower threshold when we know the index is bare leaves.
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

    # If categories_list is bare leaf names (no path separators), the TF-IDF
    # index will produce near-zero similarities for most product names.
    # Prefer full paths from code_to_path if available.
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
        # Index was built on bare leaves (or before _index_has_full_paths was added)
        # but we now have full paths available — rebuild for better accuracy.
        logger.info('[WrongCat] Rebuilding TF-IDF index with full paths')
        engine._tfidf_built = False
        engine.build_tfidf_index(_effective_cats)

    # Guard: bare-leaf TF-IDF produces too many false positives to be useful.
    # Category leaf names like 'Cases', 'High-top', 'Leather' are too short
    # and ambiguous to reliably match against product names via cosine similarity.
    # Only run wrong-category detection when we have full paths (from category_map.xlsx).
    if not getattr(engine, '_index_has_full_paths', False):
        logger.warning(
            '[WrongCat] Skipping wrong-category detection: TF-IDF index was built '
            'on bare leaf category names (no category_map.xlsx full paths available). '
            'Ensure category_map.xlsx is present and loading correctly.'
        )
        return pd.DataFrame(columns=data.columns)

    # CRITICAL: Feed the engine the JSON rules so it can use them!
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
    
    # Pre-build a leaf->full_path cache so each product row doesn't scan all paths
    leaf_to_full_path = {}
    if code_to_path:
        for full_path in code_to_path.values():
            for sep in ('/', '>'):
                if sep in full_path:
                    leaf = full_path.split(sep)[-1].strip().lower()
                    break
            else:
                leaf = full_path.strip().lower()
            # First match wins — most specific path for this leaf
            if leaf not in leaf_to_full_path:
                leaf_to_full_path[leaf] = full_path

    flagged_indices = []
    comment_map = {}
    kw_map = engine.build_keyword_to_category_mapping()

    # ── Resolution diagnostics (logged once per run) ──────────────────────────
    _diag_logged = 0
    _has_code_col = 'CATEGORY_CODE' in d.columns
    logger.info(f'[WrongCat] code_to_path size={len(code_to_path)}, '
                f'cat_path_to_code size={len(cat_path_to_code)}, '
                f'leaf_cache size={len(leaf_to_full_path)}, '
                f'has_code_col={_has_code_col}')

    for idx, row in d.iterrows():
        current_cat = row['_cat_clean']
        name = row['_name_clean']
        
        if not current_cat or current_cat.lower() in ('nan', 'none', ''):
            continue

        if 'miscellaneous' in current_cat.lower():
            flagged_indices.append(idx)
            comment_map[idx] = "Category is 'Miscellaneous'"
            continue

        # ---> CRITICAL FIX: The automated scanner now uses the JSON boost!
        predicted = engine.get_category_with_boost(name)
        
        # Fallback to standard tf-idf if boost rejected it for low confidence
        if not predicted:
            predicted = engine.get_category_with_fallback(name, kw_map, categories_list)

        # If predicted is a bare leaf name, resolve it to its full path via
        # code_to_path so segment comparison works correctly.
        if predicted and code_to_path and '/' not in predicted and '>' not in predicted:
            predicted = leaf_to_full_path.get(predicted.strip().lower(), predicted)
        
        if predicted and predicted.lower() != current_cat.lower():
            def get_top(path):
                """Return the top-level category segment, handling / and > separators."""
                for sep in ('/', '>'):
                    if sep in path:
                        return path.split(sep)[0].strip().lower()
                return path.strip().lower()

            def get_leaf(path):
                """Return the leaf segment, handling / and > separators."""
                for sep in ('/', '>'):
                    if sep in path:
                        return path.split(sep)[-1].strip().lower()
                return path.strip().lower()

            p_leaf = get_leaf(predicted)
            c_leaf = get_leaf(current_cat)

            # Skip if the current leaf already appears anywhere in the predicted path
            # e.g. current='Bluetooth Speakers', predicted='Electronics / Audio / Bluetooth Speakers'
            if c_leaf in predicted.lower():
                continue

            # Resolve current category to its full path using code_to_path so we
            # can compare top-level parents.
            # e.g. current='Smart Watches' -> 'Phones & Tablets / ... / Smart Watches'
            current_full = current_cat
            _resolution_method = 'unresolved'
            if code_to_path:
                # 1. Try resolving via CATEGORY_CODE directly (most reliable)
                row_code = str(row.get('CATEGORY_CODE', '')).strip().split('.')[0]
                if row_code and row_code in code_to_path:
                    current_full = code_to_path[row_code]
                    _resolution_method = f'code({row_code})'
                else:
                    # 2. Try cat_path_to_code lookup
                    code = cat_path_to_code.get(current_cat.lower(), '')
                    if code and code in code_to_path:
                        current_full = code_to_path[code]
                        _resolution_method = f'cat_path_to_code({code})'
                    else:
                        # 3. Use pre-built leaf cache
                        resolved = leaf_to_full_path.get(current_cat.strip().lower())
                        if resolved:
                            current_full = resolved
                            _resolution_method = 'leaf_cache'
                        # else stays as bare leaf — log it
            if _diag_logged < 10:
                logger.info(f'[WrongCat] resolution: cat={current_cat!r} '
                            f'row_code={str(row.get("CATEGORY_CODE","")).strip()!r} '
                            f'method={_resolution_method} '
                            f'current_full={current_full!r}')
                _diag_logged += 1

            # ── Segment-similarity suppression ────────────────────────────────────
            # Suppress if both paths share at least 2 leading segments.
            # e.g. 'Phones & Tablets / Accessories / Smart Watches' vs
            #      'Phones & Tablets / Accessories / Smart Watch Cables'
            # → share 2 levels → suppress (same sub-family, not a wrong category).
            def get_segments(path, n):
                """Return the first n segments of a path as a lowercase tuple."""
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

            # Pre-compute leaf/top values used by both suppression checks below
            c_leaf_lower = current_cat.strip().lower()
            c_full_lower = current_full.strip().lower()
            p_top_lower = get_top(predicted).strip().lower()

            # ── Same-domain suppression ───────────────────────────────────────────
            # When code_to_path can't resolve a bare leaf name to its full path,
            # the segment check can't fire. This dict maps top-level domain names
            # to their known sub-category leaf names so we can suppress same-domain
            # false positives without needing code_to_path at all.
            _SAME_DOMAIN_CATEGORIES = {
                # Each entry: top-level domain -> set of leaf category names that
                # GENUINELY BELONG to that domain. Only suppress if the current
                # category is a legitimate sub-category of the predicted domain.
                # Do NOT add entries just because they were false-positived into
                # a domain — that causes over-suppression.

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
                    'face primer',  # facial massager word match
                    'hairpieces', 'wrinkle & anti-aging devices', 'bubble bath',
                    'body wash', 'shaving creams', 'soaps', 'washes', 'body',
                    'face', 'lips', 'gels', 'essential oils', 'detox & cleanse',
                    'health & fitness', 'body scrubs', 'nail care', 'eye care',
                    'feminine care', 'oral care', 'medical supplies',
                },

                'home & office': {
                    # Books/media in H&O
                    'bestselling books', 'faith & spirituality',
                    # Sets & kits that are home-related
                    'sets & kits',
                    # Medical/support items shelved in H&O
                    'medical support hose',
                    # Toys shelved under H&O sub-paths (kids bathroom etc)
                    'push & pull toys', 'stacking & nesting toys',
                    # Women's clothing sub-sections sometimes shelved in H&O
                    "women's",
                    # Kitchen appliances — genuinely in H&O
                    'freezers', 'food processors', 'mixers & blenders', 'rice cookers',
                    'deep fryers', 'air fryers', 'cookers', 'microwave ovens',
                    'electric pressure cookers', 'pressure cookers', 'hot pots',
                    'waffle makers', 'toasters', 'kettles', 'coffee makers',
                    # Home tools/cleaning
                    'vacuum cleaners', 'wet & dry vacuums', 'bagless vacuum cleaner',
                    'washing machines', 'dishwashers',
                    # Furniture/storage genuinely in H&O
                    'standing shelf units', 'coat racks',
                    # Arts/crafts genuinely in H&O
                    'printer cutters', 'art set', 'canvas boards & panels',
                    # Kitchen tools genuinely in H&O
                    'kitchen utensils & gadgets', 'kitchen storage & organization accessories',
                    'stemmed water glasses', 'whisks', 'wastebasket bags',
                    # Bedding/home decor genuinely in H&O
                    'bedding sets', 'curtain panels', 'duvet covers', 'mosquito net',
                    # Small appliances
                    'usb fans',
                    # Home improvement
                    'sprayers', 'security & filtering',
                },

                'electronics': {
                    # Audio genuinely in Electronics
                    'bluetooth speakers', 'bluetooth headsets', 'earphones & headsets',
                    'portable bluetooth speakers', 'sound bars', 'headphone amplifiers',
                    'earbud headphones', 'headphone extension cables',
                    'wireless lavalier microphones',
                    # Video/display
                    'smart tvs', 'overhead projectors',
                    # Accessories genuinely in Electronics
                    'ceiling fans', 'ceiling fan light kits', 'usb fans',
                    # Remote controls
                    'tv remote controls', 'remote controls',
                    # Portable electronics
                    'gadgets',
                },

                'phones & tablets': {
                    # Accessories genuinely in P&T
                    'chargers', 'earbud headphones', 'rubber strap',
                    'electrical device mounts', 'earphones & headsets',
                    # Phones genuinely in P&T
                    'cell phones', 'android phones', 'smartphones',
                    'flip cases', 'cases', 'screen protectors',
                },

                'fashion': {
                    # Footwear genuinely in Fashion
                    'sandals', 'sneakers', 'slippers', 'shoes', 'rain boots', 'boots',
                    # Clothing genuinely in Fashion
                    'casual dresses', 'hats & caps', 'briefs', 'thongs', 'socks',
                    'unisex fabrics', 'stockings', 'polos', 'bras', 'underwear',
                    't-shirts', 'shirts', 'outerwear', 'clothing', 'dresses',
                    'jackets', 'coats', 'jeans',
                    # Accessories genuinely in Fashion
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
                    # Only suppress genuinely-Grocery sub-categories — we WANT
                    # to flag products put in wrong Grocery sub-paths
                    'standard batteries',  # batteries incorrectly in Grocery
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

            # ── Cross-domain noise suppression ────────────────────────────────────
            # Some product names contain incidental words (colors, materials, feature
            # keywords) that pull TF-IDF toward completely unrelated domains.
            # Block known noisy cross-domain leaps here.
            _CROSS_DOMAIN_BLOCKS = [
                # (current_leaf_keywords, forbidden_predicted_top_prefixes)

                # Supplements/medicine must not go to Phones & Tablets ("tablet" = pill)
                ({'supplements', 'tablets', 'capsules', 'vitamins', 'syrup', 'herbal',
                  'herbs', 'strips', 'milk substitutes'},
                 {'phones & tablets', 'electronics', 'automobile',
                  'industrial & scientific', 'sporting goods'}),

                # Clothing/Fashion must not go to Grocery, Sporting Goods, or Automobile
                ({'fashion', 'clothing', 'outerwear', 'apparel', 'shoes', 'footwear',
                  'sneakers', 'slippers', 'socks', 'polos', 'bras', 'underwear',
                  't-shirts', 'shirts', 'dresses', 'jackets', 'coats', 'jeans',
                  'sandals', 'rain boots', 'boots', 'stockings'},
                 {'grocery', 'industrial & scientific', 'automobile',
                  'sporting goods', 'electronics', 'home & office', 'pet supplies'}),

                # Electronics/Audio/Phones must not bleed into unrelated domains
                ({'electronics', 'cell phones', 'bluetooth speakers', 'bluetooth headsets',
                  'earphones', 'headsets', 'smart watches', 'wrist watches', 'tv remote',
                  'remote controls', 'wi-fi', 'dongles', 'power banks', 'earbuds',
                  'headphones', 'laptops', 'cameras', 'speakers', 'portable bluetooth'},
                 {'grocery', 'automobile', 'industrial & scientific',
                  'garden & outdoors', 'sporting goods', 'fashion', 'pet supplies'}),

                # Watches/clocks must not go to Fashion accessories or Sporting Goods
                ({'wrist watches', "women's watches", "men's watches", 'kids watches',
                  'smart watches', 'wall clocks', 'alarm clocks'},
                 {'fashion', 'sporting goods', 'grocery', 'automobile'}),

                # Health/Beauty/Personal care must not bleed into Grocery or unrelated
                ({'health', 'beauty', 'skin care', 'creams', 'makeup', 'foundation',
                  'heating pads', 'salon & spa', 'salon', 'spa', 'massage', 'medical',
                  'shaving gels', 'hair sprays', 'eau de parfum', 'fragrance', 'perfume',
                  'sets & kits'},
                 {'grocery', 'industrial & scientific', 'sporting goods',
                  'automobile', 'phones & tablets', 'toys & games', 'pet supplies'}),

                # Home/Kitchen/Furniture must not bleed into Grocery, Sporting Goods,
                # or Automobile
                ({'home', 'kitchen', 'storage', 'cleaning', 'toilet', 'coat racks',
                  'sewing machines', 'pressure cookers', 'electric pressure cookers',
                  'cookers', 'christian books', 'books', 'printer cutters', 'sprayers',
                  'art set', 'security & filtering'},
                 {'grocery', 'sporting goods', 'automobile',
                  'industrial & scientific', 'garden & outdoors'}),

                # Baby/Kids play equipment must not go to Garden or Sporting Goods
                ({'outdoor safety', 'play yard', 'baby', 'strollers', 'nursery'},
                 {'garden & outdoors', 'sporting goods', 'automobile'}),

                # Same-domain false positives: sub-categories of the same domain
                # e.g. Salon & Spa Chairs -> H&B/Massage Tools, Cell Phones -> P&T/SIM Trays,
                # Pressure Cookers -> H&O/Pressure Cooker Parts
                # These are handled by the segment check using code_to_path,
                # but as a safety net if code_to_path isn't available:
                ({'salon & spa chairs', 'massage chairs'},
                 {'health & beauty'}),
                ({'cell phones', 'earphones & headsets'},
                 {'phones & tablets'}),
                ({'pressure cookers', 'electric pressure cookers'},
                 {'home & office'}),

                # Creams/Strips/Supplements must not bleed into unrelated domains
                ({'creams', 'strips', 'supplements', 'creams & moisturizers'},
                 {'sporting goods', 'automobile', 'grocery',
                  'phones & tablets', 'industrial & scientific'}),

                # Bluetooth Headsets/Remote Controls are sub-items of Electronics/P&T
                ({'bluetooth headsets', 'tv remote controls', 'remote controls',
                  'android phones', 'musicals'},
                 {'sporting goods', 'grocery', 'automobile', 'garden & outdoors',
                  'industrial & scientific', 'fashion', 'pet supplies'}),

                # Books must not go to Office Electronics or unrelated domains
                ({'christian books & bibles', 'motivational & self-help',
                  'business & economics'},
                 {'home & office', 'industrial & scientific', 'automobile',
                  'grocery', 'sporting goods'}),

                # Kitchen appliances/tools must not go to Automobile or Sporting Goods
                ({'freezers', 'mixers & blenders', 'food processors', 'rice cookers',
                  'bakeware sets', 'utensils', 'printer cutters', 'art set',
                  'push & pull toys'},
                 {'automobile', 'sporting goods', 'grocery',
                  'industrial & scientific', 'garden & outdoors'}),

                # Umbrellas must not bleed into Fashion sub-items
                ({'stick umbrellas', 'umbrellas'},
                 {'fashion', 'grocery', 'automobile', 'sporting goods'}),

                # Bags/backpacks must not go to Electronics camera accessories
                ({'backpacks', 'camping backpacks', 'bags'},
                 {'electronics', 'automobile', 'industrial & scientific'}),

                # Video/digital games must not go to H&B or unrelated domains
                ({'digital games', 'ps 5 games', 'ps4 games', 'xbox games'},
                 {'health & beauty', 'grocery', 'automobile',
                  'industrial & scientific', 'fashion'}),

                # Hair/fabric dyes must not go to Toys tie-dye kits
                ({'dyes', 'hair dye', 'fabric dye'},
                 {'toys & games', 'grocery', 'automobile', 'industrial & scientific'}),
            ]
            blocked = False
            for current_kws, forbidden_tops in _CROSS_DOMAIN_BLOCKS:
                # Check if current category matches this block's domain
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

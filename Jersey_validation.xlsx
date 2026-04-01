"""
fix_app.py — Run this script once to patch your app.py:

    python fix_app.py app.py

It applies all four blocks from the instructions and removes the
duplicate `except` clause that causes the SyntaxError.
"""

import sys
import re

def apply_fixes(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    fixes_applied = []

    # ------------------------------------------------------------------
    # FIX 1 (Block 2): Ensure validations list has full kwargs for
    # "Wrong Category".  The old version may have an incomplete dict.
    # ------------------------------------------------------------------
    old_wc = '''        ("Wrong Category", check_miscellaneous_category, {
            'categories_list': support_files.get('categories_names_list', []),
            'compiled_rules': st.session_state.get('compiled_json_rules', {}),
            'cat_path_to_code': support_files.get('cat_path_to_code', {}),
            'code_to_path': support_files.get('code_to_path', {}),
        }),'''
    new_wc = '''        ("Wrong Category", check_miscellaneous_category, {
            'categories_list':  support_files.get('categories_names_list', []),
            'compiled_rules':   st.session_state.get('compiled_json_rules', {}),
            'cat_path_to_code': support_files.get('cat_path_to_code', {}),
            'code_to_path':     support_files.get('code_to_path', {}),
        }),'''
    # (This is cosmetic only if already present — safe no-op either way.)
    if old_wc in content:
        content = content.replace(old_wc, new_wc, 1)
        fixes_applied.append("Block 2: validations list kwargs normalised")

    # ------------------------------------------------------------------
    # FIX 2 (Block 3 & 4 + syntax error): Replace the old
    #   predicted = _engine.get_category_with_boost(name, ...)
    # calls AND remove the stray duplicate `except` block that causes
    # a SyntaxError.
    # ------------------------------------------------------------------

    # Pattern that appears TWICE in the file (once in bulk_approve_dialog,
    # once in render_flag_expander).  Both use the old signature.
    old_boost_call = (
        '                            predicted = _engine.get_category_with_boost'
        '(name, st.session_state.compiled_json_rules)\n'
    )
    new_boost_call = (
        '                            _engine.set_compiled_rules('
        'st.session_state.get(\'compiled_json_rules\', {}))\n'
        '                            predicted = _engine.get_category_with_boost(name)\n'
    )
    count = content.count(old_boost_call)
    if count:
        content = content.replace(old_boost_call, new_boost_call)
        fixes_applied.append(f"Blocks 3 & 4: replaced old get_category_with_boost call ({count}x)")

    # ------------------------------------------------------------------
    # FIX 3 (Syntax error): Remove the duplicate / orphaned `except`
    # clause in bulk_approve_dialog.  It appears right after the first
    # correct except and has no matching try.
    # ------------------------------------------------------------------
    bad_dup = (
        '                except Exception as _le:\n'
        '                    logger.warning("Wrong Category approval learning failed: %s", _le)\n'
        '                except Exception as _le:\n'
        '                    logger.warning("Wrong Category approval learning failed: %s", _le)\n'
    )
    good_single = (
        '                except Exception as _le:\n'
        '                    logger.warning("Wrong Category approval learning failed: %s", _le)\n'
    )
    if bad_dup in content:
        content = content.replace(bad_dup, good_single, 1)
        fixes_applied.append("Fix 3: removed duplicate orphaned except in bulk_approve_dialog")
    else:
        # Also try with the comment lines that appear in between
        bad_dup2 = (
            '                except Exception as _le:\n'
            '                    logger.warning("Wrong Category approval learning failed: %s", _le)\n'
            '                except Exception as _le:\n'
            '                    logger.warning("Wrong Category approval learning failed: %s", _le)'
        )
        if bad_dup2 in content:
            content = content.replace(bad_dup2, good_single.rstrip('\n'), 1)
            fixes_applied.append("Fix 3 (alt): removed duplicate orphaned except in bulk_approve_dialog")

    # ------------------------------------------------------------------
    # Validate: quick syntax check before writing
    # ------------------------------------------------------------------
    try:
        compile(content, path, 'exec')
        print("✅ Syntax check passed.")
    except SyntaxError as e:
        print(f"⚠️  Syntax error still present at line {e.lineno}: {e.msg}")
        print("   File NOT saved.  Please report the line number above.")
        return

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

    if fixes_applied:
        for fix in fixes_applied:
            print(f"  ✔  {fix}")
        print(f"\n✅ {len(fixes_applied)} fix(es) applied and saved to {path}")
    else:
        print("ℹ️  No matching patterns found — file may already be patched, or uses different formatting.")
        print("   Check the manual instructions below.\n")
        print_manual_instructions()


def print_manual_instructions():
    print("""
Manual fix instructions
=======================

BLOCK 3 & 4  —  find both occurrences of this line:
    predicted = _engine.get_category_with_boost(name, st.session_state.compiled_json_rules)

Replace each one with:
    _engine.set_compiled_rules(st.session_state.get('compiled_json_rules', {}))
    predicted = _engine.get_category_with_boost(name)

SYNTAX ERROR fix  —  find this block (inside bulk_approve_dialog):
    except Exception as _le:
        logger.warning("Wrong Category approval learning failed: %s", _le)
    except Exception as _le:                      <-- DELETE THIS LINE
        logger.warning("Wrong Category approval learning failed: %s", _le)  <-- AND THIS

Keep only ONE of the two except/logger lines.
""")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python fix_app.py <path/to/app.py>")
        sys.exit(1)
    apply_fixes(sys.argv[1])

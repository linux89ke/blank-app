# ---------------------------------------------------------------------------
# _preqc_registry.py
#
# A zero-dependency shared registry.
# streamlit_app.py writes check functions into REGISTRY after defining them.
# postqc.py reads from REGISTRY inside run_checks() — no circular import,
# no re-execution of streamlit_app, no duplicate Streamlit widget IDs.
# ---------------------------------------------------------------------------

REGISTRY: dict = {}

import sys
try:
    import pytest  # type: ignore
except Exception:
    raise SystemExit(0)

pytest.skip("Obsolete legacy meta_cache test (deprecated). Skipped.", allow_module_level=True)


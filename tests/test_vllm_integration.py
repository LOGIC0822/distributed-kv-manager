import sys
try:
    import pytest  # type: ignore
except Exception:
    raise SystemExit(0)

pytest.skip("Obsolete vLLM integration test (deprecated). Skipped.", allow_module_level=True)

# Legacy content removed.
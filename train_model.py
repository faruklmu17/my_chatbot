# train_model.py
import requests
import json
import joblib
from datetime import datetime, timezone
from typing import Iterator, Tuple, Optional, Any, Dict

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# ✅ Default GitHub raw link for testing
GITHUB_JSON_URL = "https://raw.githubusercontent.com/faruklmu17/browser_extension_test/refs/heads/main/tests/test-results.json"

def fetch_data() -> Dict[str, Any]:
    """Fetches JSON data from the GitHub raw URL."""
    resp = requests.get(GITHUB_JSON_URL, timeout=20)
    if resp.status_code == 200:
        return resp.json()
    raise RuntimeError(f"Failed to fetch JSON. Status code: {resp.status_code}")

def iter_tests(data) -> Iterator[Tuple[str, str, str, str, Optional[str], bool]]:
    """
    Yield (suite_title, spec_title, test_title, final_status, project_name, was_flaky)
    Supports:
      - suites -> specs -> tests -> results (newer Playwright JSON)
      - suites -> tests -> results (flat inside suite)
    """
    for suite in data.get("suites", []):
        suite_title = suite.get("title") or ""
        # pattern A: specs -> tests
        for spec in suite.get("specs", []):
            spec_title = spec.get("title") or spec.get("file") or ""
            for test in spec.get("tests", []):
                final_status, project, flaky = _final_status_project_flaky(test)
                yield (suite_title, spec_title, test.get("title", ""), final_status, project, flaky)

        # pattern B: tests directly under suite
        for test in suite.get("tests", []):
            final_status, project, flaky = _final_status_project_flaky(test)
            yield (suite_title, "", test.get("title", ""), final_status, project, flaky)

def _final_status_project_flaky(test: Dict[str, Any]) -> Tuple[str, Optional[str], bool]:
    """Return final status (last result), projectName, flaky."""
    results = test.get("results") or []
    if not results:
        return ("unknown", test.get("projectName"), False)
    last = results[-1]
    status = last.get("status", "unknown")
    flaky = bool(last.get("flaky") or test.get("flaky"))
    project = test.get("projectName") or last.get("projectName")
    return (status, project, flaky)

# ---------- NEW: last-run capable iterator ----------

def _parse_ts(value: Any) -> Optional[datetime]:
    """
    Robust timestamp parser.
    Accepts ISO8601 strings (with/without Z), or epoch seconds/ms in int/float/str.
    Returns timezone-aware UTC datetime.
    """
    if value is None:
        return None
    # epoch-like (seconds or ms)
    try:
        if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().isdigit()):
            v = float(value)
            # heuristically treat large numbers as ms
            if v > 1e12:
                v /= 1000.0
            return datetime.fromtimestamp(v, tz=timezone.utc)
    except Exception:
        pass

    if isinstance(value, str):
        s = value.strip()
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None

def _best_result_time(result: Dict[str, Any]) -> Optional[datetime]:
    """
    Find the most reliable start time field on a single result object.
    Common fields: startTime, startTimeEpoch, startTimeUnix, startTimeMS
    """
    candidates = [
        result.get("startTime"),
        result.get("startTimeEpoch"),
        result.get("startTimeUnix"),
        result.get("startTimeMs"),
        result.get("startTimeMS"),
        result.get("startTimeMilliseconds"),
    ]
    for c in candidates:
        dt = _parse_ts(c)
        if dt:
            return dt
    # Some JSONs only have `startTime` at the test level; fallback handled by caller.
    return None

def iter_tests_with_time(data) -> Iterator[Tuple[str, str, str, str, Optional[str], bool, Optional[datetime]]]:
    """
    Yield (suite_title, spec_title, test_title, final_status, project_name, was_flaky, start_time_utc)
    start_time_utc is the start time of the LAST RETRY (i.e., final attempt), UTC tz-aware.
    """
    for suite in data.get("suites", []):
        suite_title = suite.get("title") or ""
        # pattern A
        for spec in suite.get("specs", []):
            spec_title = spec.get("title") or spec.get("file") or ""
            for test in spec.get("tests", []):
                final_status, project, flaky = _final_status_project_flaky(test)
                start_dt = _extract_last_attempt_time(test)
                yield (suite_title, spec_title, test.get("title", ""), final_status, project, flaky, start_dt)
        # pattern B
        for test in suite.get("tests", []):
            final_status, project, flaky = _final_status_project_flaky(test)
            start_dt = _extract_last_attempt_time(test)
            yield (suite_title, "", test.get("title", ""), final_status, project, flaky, start_dt)

def _extract_last_attempt_time(test: Dict[str, Any]) -> Optional[datetime]:
    """
    For a test with retries, take the last result's start time.
    If not found on results, try test-level fields as a fallback.
    """
    results = test.get("results") or []
    if results:
        last = results[-1]
        dt = _best_result_time(last)
        if dt:
            return dt
    # Fallback to test-level
    test_level_candidates = [
        test.get("startTime"),
        test.get("startTimeEpoch"),
        test.get("startTimeUnix"),
        test.get("startTimeMs"),
        test.get("startTimeMS"),
    ]
    for c in test_level_candidates:
        dt = _parse_ts(c)
        if dt:
            return dt
    return None

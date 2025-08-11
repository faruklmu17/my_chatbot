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
    """Fetch the Playwright JSON from GitHub (or your endpoint)."""
    resp = requests.get(GITHUB_JSON_URL, timeout=20)
    if resp.status_code == 200:
        return resp.json()
    raise RuntimeError(f"Failed to fetch JSON. Status code: {resp.status_code}")

# ---------------- Core test iteration (compatible with earlier code) ----------------

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
    """Return final status (last retry), projectName, flaky."""
    results = test.get("results") or []
    if not results:
        return ("unknown", test.get("projectName"), bool(test.get("flaky")))
    last = results[-1]
    status = last.get("status", "unknown")
    flaky = bool(last.get("flaky") or test.get("flaky"))
    project = test.get("projectName") or last.get("projectName")
    return (status, project, flaky)

# ---------------- Time parsing utilities (NEW) ----------------

def _parse_ts(value: Any) -> Optional[datetime]:
    """
    Robust timestamp parser -> timezone-aware UTC datetime.
    Accepts:
      - ISO8601 strings, with or without Z
      - epoch seconds or milliseconds (int/float/str)
    """
    if value is None:
        return None

    # epoch number or numeric string
    try:
        if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().replace(".", "", 1).isdigit()):
            v = float(value)
            if v > 1e12:  # likely milliseconds -> seconds
                v /= 1000.0
            return datetime.fromtimestamp(v, tz=timezone.utc)
    except Exception:
        pass

    # ISO8601
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
    Pick the most reliable time field from a single result object.
    """
    for key in ("startTime", "startTimeEpoch", "startTimeUnix", "startTimeMs", "startTimeMS", "startTimeMilliseconds"):
        dt = _parse_ts(result.get(key))
        if dt:
            return dt
    return None

def _extract_last_attempt_time(test: Dict[str, Any]) -> Optional[datetime]:
    """
    For a test with retries, take the LAST result's start time.
    If not found on results, try test-level time fields as a fallback.
    """
    results = test.get("results") or []
    if results:
        dt = _best_result_time(results[-1])
        if dt:
            return dt

    for key in ("startTime", "startTimeEpoch", "startTimeUnix", "startTimeMs", "startTimeMS"):
        dt = _parse_ts(test.get(key))
        if dt:
            return dt

    return None

def iter_tests_with_time(data) -> Iterator[Tuple[str, str, str, str, Optional[str], bool, Optional[datetime]]]:
    """
    Yield (suite_title, spec_title, test_title, final_status, project_name, was_flaky, start_time_utc)
    start_time_utc is the start time of the LAST RETRY (final attempt), tz-aware UTC.
    """
    for suite in data.get("suites", []):
        suite_title = suite.get("title") or ""
        # pattern A
        for spec in suite.get("specs", []):
            spec_title = spec.get("title") or spec.get("file") or ""
            for test in spec.get("tests", []):
                final_status, project, flaky = _final_status_project_flaky(test)
                yield (
                    suite_title,
                    spec_title,
                    test.get("title", ""),
                    final_status,
                    project,
                    flaky,
                    _extract_last_attempt_time(test),
                )
        # pattern B
        for test in suite.get("tests", []):
            final_status, project, flaky = _final_status_project_flaky(test)
            yield (
                suite_title,
                "",
                test.get("title", ""),
                final_status,
                project,
                flaky,
                _extract_last_attempt_time(test),
            )

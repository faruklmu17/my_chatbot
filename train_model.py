# train_model.py
import requests
import joblib
from typing import Dict, Any, List, Tuple, Iterator
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# ✅ Default GitHub raw link for testing
GITHUB_JSON_URL = "https://raw.githubusercontent.com/faruklmu17/browser_extension_test/refs/heads/main/tests/test-results.json"

# ---- Status normalization ----
FAIL_STATUSES = {"failed", "timedout", "timed_out", "interrupted"}
PASS_STATUSES = {"passed"}
SKIP_STATUSES = {"skipped"}

def _status_str(s: str) -> str:
    if not s:
        return "unknown"
    s = s.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    # map common variants
    if s in {"timedout", "timeout", "timedouterror"}:
        return "timedout"
    if s in {"pass", "ok", "success"}:
        return "passed"
    if s in {"skip", "skipping"}:
        return "skipped"
    if s in {"fail", "error"}:
        return "failed"
    return s

# ---- JSON walkers that are defensive about schema ----
def _attempts_from_test(test: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract attempts from a test object across common Playwright reporter shapes.
    Returns a list of {status, duration_ms, errors}.
    """
    attempts: List[Dict[str, Any]] = []

    for key in ("retries", "attempts", "results"):
        arr = test.get(key)
        if isinstance(arr, list) and arr:
            for a in arr:
                status = _status_str(a.get("status") or a.get("outcome") or test.get("status") or "")
                attempts.append({
                    "status": status,
                    "duration_ms": a.get("duration") or a.get("durationMs"),
                    "errors": a.get("errors") or a.get("error") or [],
                })
            break

    if not attempts:
        status = _status_str(test.get("status") or test.get("outcome") or "")
        attempts.append({
            "status": status or "unknown",
            "duration_ms": test.get("duration") or test.get("durationMs"),
            "errors": test.get("errors") or test.get("error") or [],
        })

    # Fill unknowns defensively
    for a in attempts:
        if not a["status"]:
            a["status"] = "unknown"
    return attempts

def _walk_playwright(data: Dict[str, Any]):
    """
    Yields (suite_title, spec_title, test_obj) for each test in the report.
    Supports nested suites/specs and alternate 'results' layout.
    """
    def _iter_suite(suite: Dict[str, Any]):
        suite_title = suite.get("title") or suite.get("name") or suite.get("file") or ""
        # specs with tests
        for spec in suite.get("specs", []):
            spec_title = spec.get("title") or spec.get("file") or ""
            for test in spec.get("tests", []):
                yield suite_title, spec_title, test
        # tests directly under suite
        for test in suite.get("tests", []):
            yield suite_title, suite_title, test
        # nested
        for child in suite.get("suites", []):
            yield from _iter_suite(child)

    if isinstance(data.get("suites"), list):
        for suite in data["suites"]:
            yield from _iter_suite(suite)
        return

    # Fallback reporters
    for r in data.get("results", []):
        suite_title = (r.get("suite") or {}).get("title") or (r.get("suite") or {}).get("name") or ""
        spec_title = r.get("file") or r.get("title") or ""
        for test in r.get("tests", []):
            yield suite_title, spec_title, test

def _guess_project(test: Dict[str, Any]) -> str:
    return test.get("projectName") or test.get("project") or test.get("projectId") or "default"

def _normalize_tests(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build a dict keyed by a stable composite id with rich info per test.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for suite_title, spec_title, test in _walk_playwright(data):
        title = test.get("title") or ""
        project = _guess_project(test)
        attempts = _attempts_from_test(test)
        final_status = attempts[-1]["status"] if attempts else "unknown"
        failed_once = any(a["status"] in FAIL_STATUSES for a in attempts)
        is_flaky = failed_once and (final_status in PASS_STATUSES)

        test_id = test.get("id") or f"{suite_title}::{spec_title}::{title}::{project}"
        out[test_id] = {
            "suite": suite_title or "Unknown suite",
            "spec": spec_title or "Unknown spec",
            "title": title or "Unknown test",
            "project": project,
            "attempts": attempts,
            "final_status": final_status,
            "failed_once": failed_once,
            "is_flaky": is_flaky,
        }
    return out

# ---- Public API used by app.py ----
def fetch_data() -> Dict[str, Any]:
    """Fetches JSON data from the GitHub raw URL."""
    resp = requests.get(GITHUB_JSON_URL, timeout=20)
    if resp.status_code == 200:
        return resp.json()
    raise RuntimeError(f"Failed to fetch JSON. Status code: {resp.status_code}")

def iter_tests_with_attempts(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Yields rich dicts for each test:
      {suite, spec, title, project, attempts, final_status, failed_once, is_flaky}
    """
    normalized = _normalize_tests(data)
    return iter(normalized.values())

def iter_tests(data: Dict[str, Any]):
    """
    Backward‑compatible iterator used by older app.py versions.
    Yields: (suite_title, spec_title, test_title, final_status, project_name, was_flaky)
    """
    for t in iter_tests_with_attempts(data):
        yield (
            t["suite"],
            t["spec"],
            t["title"],
            t["final_status"],
            t["project"],
            t["is_flaky"],
        )

# ---- Naive Bayes trainer (unchanged interface, better text) ----
def train_bot():
    """Fetch Playwright JSON, generate Q&A pairs, train Naive Bayes, and save artifacts."""
    data = fetch_data()
    questions: List[str] = []
    answers: List[str] = []

    for suite_title, spec_title, test_title, final_status, project, _ in iter_tests(data):
        fs = _status_str(final_status)
        # Q/A per test
        questions.append(f"What was the result of '{test_title}'?")
        answers.append(f"The test '{test_title}' in '{suite_title}' {fs}.")

        # Project variant
        questions.append(f"Did '{test_title}' pass on {project}?")
        answers.append("Yes, it passed." if fs in PASS_STATUSES else "No, it did not pass.")

        # Fail variant
        questions.append(f"Did '{test_title}' fail?")
        answers.append("Yes, it failed." if fs in FAIL_STATUSES else "No, it did not fail.")

    if not questions:
        raise ValueError("No questions generated. The JSON structure may be unexpected. Inspect the file shape.")

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(questions)
    y = list(range(len(answers)))

    model = MultinomialNB()
    model.fit(X, y)

    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(answers, "answers.pkl")
    print("✅ Model trained and saved successfully.")

if __name__ == "__main__":
    train_bot()
import requests
import json
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# ✅ Default GitHub raw link for testing
GITHUB_JSON_URL = "https://raw.githubusercontent.com/faruklmu17/browser_extension_test/refs/heads/main/tests/test-results.json"

def fetch_data():
    """Fetches JSON data from the GitHub raw URL."""
    resp = requests.get(GITHUB_JSON_URL)
    if resp.status_code == 200:
        return resp.json()
    raise RuntimeError(f"Failed to fetch JSON. Status code: {resp.status_code}")

def iter_tests(data):
    """
    Yield (suite_title, spec_title, test_title, final_status, project_name, was_flaky)
    Supports:
      - suites -> specs -> tests -> results (newer Playwright JSON)
      - suites -> tests -> results (older/simple)
    """
    suites = data.get("suites", []) or []
    for suite in suites:
        suite_title = suite.get("title") or suite.get("file") or "Unknown suite"

        def analyze_test(test, spec_title_guess):
            test_title = test.get("title", "Unknown test")
            results = test.get("results", []) or []
            statuses = [r.get("status") for r in results if isinstance(r, dict)]
            final_status = statuses[-1] if statuses else (test.get("status") or "unknown")
            project_name = test.get("projectName") or test.get("projectId") or "unknown"
            # Flaky: multiple attempts and final pass after a prior fail/timeout, or explicit flag
            was_flaky = bool(
                test.get("flaky")
                or (len(statuses) > 1 and final_status == "passed" and any(s in ("failed", "timedOut") for s in statuses[:-1]))
            )
            return suite_title, spec_title_guess, test_title, final_status, project_name, was_flaky

        # A) specs under suite
        if "specs" in suite:
            for spec in suite.get("specs", []):
                spec_title = spec.get("title") or spec.get("file") or "Unknown spec"
                for test in spec.get("tests", []):
                    yield analyze_test(test, spec_title)

        # B) tests directly under suite
        if "tests" in suite:
            for test in suite.get("tests", []):
                yield analyze_test(test, suite_title)

def train_bot():
    """Fetch Playwright JSON, generate Q&A pairs, train Naive Bayes, and save artifacts."""
    data = fetch_data()

    questions, answers = [], []

    # NOTE: iter_tests returns 6 values; use the underscore to ignore was_flaky here
    for suite_title, spec_title, test_title, final_status, project, _ in iter_tests(data):
        # Q/A per test
        questions.append(f"What was the result of '{test_title}'?")
        answers.append(f"The test '{test_title}' in '{suite_title}' {final_status}.")

        # Include project-flavored variants (helps match more queries)
        questions.append(f"Did '{test_title}' pass on {project}?")
        answers.append("Yes, it passed." if final_status == "passed" else "No, it did not pass.")

        questions.append(f"Did '{test_title}' fail?")
        answers.append("Yes, it failed." if final_status == "failed" else "No, it did not fail.")

    if not questions:
        raise ValueError("No questions generated. The JSON structure may be unexpected. Inspect the file shape.")

    # Train Naive Bayes
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(questions)
    y = list(range(len(answers)))

    model = MultinomialNB()
    model.fit(X, y)

    # Save artifacts
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(answers, "answers.pkl")
    print("✅ Model trained and saved successfully.")

if __name__ == "__main__":
    train_bot()

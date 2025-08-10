# app.py
import re
import gradio as gr
import joblib
import traceback

import train_model  # we reuse fetch_data() and iter_tests()

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
ANSWERS_PATH = "answers.pkl"

model = None
vectorizer = None
answers = None

def load_artifacts():
    global model, vectorizer, answers
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    answers = joblib.load(ANSWERS_PATH)

def get_snapshot():
    """Pull the latest JSON (no retrain) and compute aggregates and lists, including flaky."""
    data = train_model.fetch_data()
    totals = {"passed": 0, "failed": 0, "skipped": 0, "flaky": 0, "unknown": 0}
    per_suite = {}
    passed_tests, failed_tests, flaky_tests = [], [], []

    for suite_title, spec_title, test_title, final_status, project, is_flaky in train_model.iter_tests(data):
        name = test_title or spec_title or "Unnamed test"
        sname = suite_title or "Unknown suite"

        # Treat flaky as its own status (does not double-count as passed)
        status = "flaky" if is_flaky else (final_status or "unknown").lower()

        totals[status] = totals.get(status, 0) + 1
        per_suite.setdefault(sname, {"passed":0,"failed":0,"skipped":0,"flaky":0,"unknown":0})
        per_suite[sname][status] = per_suite[sname].get(status, 0) + 1

        if status == "passed":
            passed_tests.append(f"{name} ({sname})")
        elif status == "failed":
            failed_tests.append(f"{name} ({sname})")
        elif status == "flaky":
            flaky_tests.append(f"{name} ({sname})")

    return totals, per_suite, passed_tests, failed_tests, flaky_tests


def handle_aggregate_intents(message: str):
    """Simple intent router for counts/lists (passed/failed/skipped/flaky). Returns a string or None."""
    import re
    msg = message.lower().strip()

    totals, per_suite, passed_tests, failed_tests, flaky_tests = get_snapshot()

    # "any flaky tests?" -> yes/no + count
    if re.search(r"\b(any\s+)?flaky tests?\b|\bflaky\?\b", msg):
        if totals.get("flaky", 0) == 0:
            return "No flaky tests."
        items = "\n".join(f"- {t}" for t in flaky_tests[:50])
        more = "" if len(flaky_tests) <= 50 else f"\n… and {len(flaky_tests)-50} more."
        return f"{totals['flaky']} flaky tests:\n{items}{more}"

    # how many passed/failed/skipped/flaky
    m = re.search(r"(how many|count|number of)\s+(tests?\s+)?(passed|failed|skipped|flaky)", msg)
    if m:
        what = m.group(3)
        return f"{totals.get(what,0)} tests {what}."

    # total tests
    if re.search(r"(how many|count|number of)\s+(tests?)\b", msg):
        total = sum(totals.values())
        return f"Total tests: {total} (passed {totals['passed']}, failed {totals['failed']}, skipped {totals['skipped']}, flaky {totals['flaky']})."

    # list failed / passed / flaky
    if re.search(r"\b(list|show|which)\b.*\b(failed tests|tests that failed|failures)\b", msg):
        if not failed_tests:
            return "No failed tests."
        items = "\n".join(f"- {t}" for t in failed_tests[:50])
        more = "" if len(failed_tests) <= 50 else f"\n… and {len(failed_tests)-50} more."
        return f"Failed tests:\n{items}{more}"

    if re.search(r"\b(list|show|which)\b.*\b(passed tests|tests that passed)\b", msg):
        if not passed_tests:
            return "No passed tests."
        items = "\n".join(f"- {t}" for t in passed_tests[:50])
        more = "" if len(passed_tests) <= 50 else f"\n… and {len(passed_tests)-50} more."
        return f"Passed tests:\n{items}{more}"

    if re.search(r"\b(list|show|which)\b.*\b(flaky tests?)\b", msg):
        if not flaky_tests:
            return "No flaky tests."
        items = "\n".join(f"- {t}" for t in flaky_tests[:50])
        more = "" if len(flaky_tests) <= 50 else f"\n… and {len(flaky_tests)-50} more."
        return f"Flaky tests:\n{items}{more}"

    # per-suite example: "how many flaky in checkout"
    m = re.search(r"(how many|count|number of)\s+(tests?\s+)?(passed|failed|skipped|flaky)\s+in\s+(.+)", msg)
    if m:
        what = m.group(3)
        suite_query = m.group(4).strip().strip("?")
        candidates = [s for s in per_suite.keys() if suite_query.lower() in s.lower()]
        if not candidates:
            return f"I couldn't find a suite matching '{suite_query}'."
        s = candidates[0]
        return f"In '{s}': {per_suite[s].get(what,0)} tests {what}."

    return None


def respond(message, history):
    # First: try aggregate intents
    try:
        agg = handle_aggregate_intents(message)
        if agg is not None:
            history = history + [(message, agg)]
            return history, gr.update(value="")
    except Exception as e:
        # don’t break chat if aggregate fails
        pass

    # Fallback: ML classifier for per-test Q&A
    try:
        X = vectorizer.transform([message])
        pred = model.predict(X)[0]
        reply = answers[pred]
    except Exception as e:
        reply = f"Error answering your question:\n{e}\n\n{traceback.format_exc()}"

    history = history + [(message, reply)]
    return history, gr.update(value="")

def do_refresh():
    """Retrain from latest GitHub JSON and hot-reload the model."""
    try:
        train_model.train_bot()
        load_artifacts()
        return "✅ Refreshed: model retrained from latest GitHub JSON."
    except Exception as e:
        return f"❌ Refresh failed: {e}"

# Init
load_artifacts()

with gr.Blocks(title="Playwright Test Bot") as demo:
    gr.Markdown("# 🤖 Playwright Test Bot\nAsk about your Playwright test results.")
    with gr.Row():
        refresh_btn = gr.Button("🔄 Refresh from GitHub & Retrain")
        refresh_status = gr.Markdown("")
    chat = gr.Chatbot(height=420)
    msg = gr.Textbox(placeholder="Try: How many tests passed?")
    send = gr.Button("Send", variant="primary")

    send.click(fn=respond, inputs=[msg, chat], outputs=[chat, msg])
    msg.submit(fn=respond, inputs=[msg, chat], outputs=[chat, msg])
    refresh_btn.click(fn=do_refresh, outputs=refresh_status)

if __name__ == "__main__":
    demo.launch()

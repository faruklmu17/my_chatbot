# app.py
import re
import gradio as gr
import joblib
import traceback
from typing import List, Dict, Optional

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
    data = train_model.fetch_data()
    totals = {"passed": 0, "failed": 0, "skipped": 0, "flaky": 0, "unknown": 0}
    per_suite = {}
    passed_tests, failed_tests, flaky_tests = [], [], []

    for suite_title, spec_title, test_title, final_status, project, is_flaky in train_model.iter_tests(data):
        name = test_title or spec_title or "Unnamed test"
        sname = suite_title or "Unknown suite"
        status = "flaky" if is_flaky else (final_status or "unknown").lower()

        totals[status] = totals.get(status, 0) + 1
        per_suite.setdefault(sname, {"passed": 0, "failed": 0, "skipped": 0, "flaky": 0, "unknown": 0})
        per_suite[sname][status] = per_suite[sname].get(status, 0) + 1

        if status == "passed":
            passed_tests.append(f"{name} ({sname})")
        elif status == "failed":
            failed_tests.append(f"{name} ({sname})")
        elif status == "flaky":
            flaky_tests.append(f"{name} ({sname})")

    return totals, per_suite, passed_tests, failed_tests, flaky_tests

def handle_aggregate_intents(message: str) -> Optional[str]:
    msg = message.lower().strip()
    totals, per_suite, passed_tests, failed_tests, flaky_tests = get_snapshot()

    if re.search(r"\b(any\s+)?flaky tests?\b|\bflaky\?\b", msg):
        if totals.get("flaky", 0) == 0:
            return "No flaky tests."
        items = "\n".join(f"- {t}" for t in flaky_tests[:50])
        more = "" if len(flaky_tests) <= 50 else f"\n… and {len(flaky_tests)-50} more."
        return f"{totals['flaky']} flaky tests:\n{items}{more}"

    m = re.search(r"(how many|count|number of)\s+(tests?\s+)?(passed|failed|skipped|flaky)", msg)
    if m:
        what = m.group(3)
        return f"{totals.get(what,0)} tests {what}."

    if re.search(r"(how many|count|number of)\s+(tests?)\b", msg):
        total = sum(totals.values())
        return f"Total tests: {total} (passed {totals['passed']}, failed {totals['failed']}, skipped {totals['skipped']}, flaky {totals['flaky']})."

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

Message = Dict[str, str]

def _append_exchange(history: Optional[List[Message]], user_msg: str, assistant_msg: str) -> List[Message]:
    history = history or []
    history = history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": assistant_msg}]
    return history

def respond(user_text: str, history_msgs: Optional[List[Message]]):
    """Gradio callback that takes user text + Chatbot messages and returns updated messages.

    With Chatbot(type="messages"), `history_msgs` is a list of {role, content} dicts.
    """
    text = (user_text or "").strip()

    # Helper to provide a concise snapshot/help when the query is vague
    def _short_help() -> str:
        totals, *_ = get_snapshot()
        total = sum(totals.values())
        lines = [
            "Here’s the current snapshot:",
            f"• Total: {total}",
            f"• Passed: {totals['passed']}",
            f"• Failed: {totals['failed']}",
            f"• Skipped: {totals['skipped']}",
            f"• Flaky: {totals['flaky']}",
            "",
            "Try: 'list failed tests', 'any flaky tests?', or 'how many passed in <suite>'.",
        ]
        return "\n".join(lines)

    # 1) If too vague like just 'tests' or 'results', show summary to avoid hallucinated per-test answers
    if re.fullmatch(r"(?i)(tests?|results?)\??", text):
        updated = _append_exchange(history_msgs, user_text, _short_help())
        return updated, gr.update(value="")

    # 2) Aggregate intents (counts/lists/suites)
    try:
        agg = handle_aggregate_intents(text)
        if agg is not None:
            updated = _append_exchange(history_msgs, user_text, agg)
            return updated, gr.update(value="")
    except Exception:
        # don't break chat if aggregate fails
        pass

    # 3) Fallback: ML classifier for per-test Q&A, with a confidence guardrail
    try:
        if model is None or vectorizer is None or answers is None:
            raise RuntimeError("Artifacts not loaded")
        X = vectorizer.transform([text])
        reply: Optional[str] = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            idx = int(proba.argmax())
            conf = float(proba[idx])
            pred = model.classes_[idx]
            reply = answers[pred]
            if conf < 0.45:
                reply = _short_help()
        else:
            pred = model.predict(X)[0]
            reply = answers[pred]
    except Exception as e:
        reply = f"Error answering your question:\n{e}\n\n{traceback.format_exc()}"

    updated = _append_exchange(history_msgs, user_text, reply)
    return updated, gr.update(value="")


def do_refresh():
    try:
        train_model.train_bot()
        load_artifacts()
        return "✅ Refreshed: model retrained from latest GitHub JSON."
    except Exception as e:
        return f"❌ Refresh failed: {e}"

load_artifacts()

with gr.Blocks(title="Playwright Test Bot") as demo:
    gr.Markdown("# 🤖 Playwright Test Bot\nAsk about your Playwright test results.")
    with gr.Row():
        refresh_btn = gr.Button("🔄 Refresh from GitHub & Retrain")
        refresh_status = gr.Markdown("")

    chat = gr.Chatbot(type="messages", height=420)
    msg = gr.Textbox(placeholder="Try: How many tests passed?")
    send = gr.Button("Send", variant="primary")

    send.click(fn=respond, inputs=[msg, chat], outputs=[chat, msg])
    msg.submit(fn=respond, inputs=[msg, chat], outputs=[chat, msg])
    refresh_btn.click(fn=do_refresh, outputs=refresh_status)

if __name__ == "__main__":
    demo.launch()

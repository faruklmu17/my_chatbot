# app.py
import re
import gradio as gr
import joblib
import traceback
from typing import List, Dict, Optional, Any

import train_model  # uses fetch_data() and iter_tests()/iter_tests_with_attempts()

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
ANSWERS_PATH = "answers.pkl"

model = None
vectorizer = None
answers = None

# ---- Normalization helpers ----
PASS = {"passed"}
SKIP = {"skipped"}
FAIL = {"failed", "timedout", "timed_out", "interrupted"}
KNOWN = PASS | SKIP | FAIL | {"unknown"}

def _norm_status(s: Optional[str]) -> str:
    if not s:
        return "unknown"
    s = s.strip().lower().replace(" ", "").replace("-", "_")
    return s if s in KNOWN else s

def _safe(s: Optional[str], default: str) -> str:
    return (s or "").strip() or default

# ---- Greeting handling ----
GREETING_REPLY = "Hello! 👋 Hope you're doing well. How can I help you with your Playwright tests today?"

def is_greeting(text: str) -> bool:
    """
    Detect common greetings/pleasantries (hi, hello, gm/gn, good morning/evening,
    how are you / how r u, hey, gd evening, etc.).
    """
    pattern = r"""
        (?:
          \b(hi|hello|hey)\b
          | \bgm\b | \bgn\b
          | \bgd\s*(morning|afternoon|evening)\b
          | \bgood\s*(morning|afternoon|evening|night)\b
          | \bhow\s*(are\s*you|r\s*u)\b
        )
    """
    return bool(re.search(pattern, text.lower(), flags=re.VERBOSE))

# ---- Load artifacts (unchanged) ----
def load_artifacts():
    global model, vectorizer, answers
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    answers = joblib.load(ANSWERS_PATH)

# ---- Snapshot that understands retries/attempts ----
def get_snapshot():
    """
    Returns:
      totals: dict
      per_suite: dict
      passed_tests: [str]
      failed_tests: [str]                  # FINAL failures only
      flaky_tests: [str]                   # failed then passed
      failed_at_least_once_names: [str]    # failed in any attempt (incl. flaky + final fails)
    """
    data = train_model.fetch_data()

    # Prefer richer iterator with attempts; fallback to older one
    use_rich = hasattr(train_model, "iter_tests_with_attempts")
    tests: List[Dict[str, Any]] = []

    if use_rich:
        tests = list(train_model.iter_tests_with_attempts(data))
    else:
        for suite_title, spec_title, test_title, final_status, project, is_flaky in train_model.iter_tests(data):
            tests.append({
                "suite": suite_title,
                "spec": spec_title,
                "title": test_title,
                "project": project,
                "attempts": [{"status": _norm_status(final_status)}],
                "final_status": _norm_status(final_status),
                "is_flaky": bool(is_flaky),
                "failed_once": _norm_status(final_status) in FAIL or bool(is_flaky),
            })

    totals = {"passed": 0, "failed": 0, "skipped": 0, "flaky": 0, "unknown": 0}
    per_suite: Dict[str, Dict[str, int]] = {}
    passed_tests, failed_tests, flaky_tests = [], [], []
    failed_at_least_once_names = []

    for t in tests:
        name = _safe(t.get("title") or t.get("spec"), "Unnamed test")
        sname = _safe(t.get("suite"), "Unknown suite")
        proj = _safe(t.get("project"), "default")

        final_status = _norm_status(t.get("final_status"))
        if not final_status:
            attempts = t.get("attempts") or []
            final_status = _norm_status((attempts[-1].get("status") if attempts else None))

        is_flaky = bool(t.get("is_flaky"))
        failed_once = bool(t.get("failed_once"))

        key = "flaky" if is_flaky else (final_status or "unknown")
        totals[key] = totals.get(key, 0) + 1
        per_suite.setdefault(sname, {"passed": 0, "failed": 0, "skipped": 0, "flaky": 0, "unknown": 0})
        per_suite[sname][key] = per_suite[sname].get(key, 0) + 1

        label = f"{name} ({sname})"
        if is_flaky:
            flaky_tests.append(label)
        if final_status in PASS:
            passed_tests.append(label)
        elif final_status in FAIL:
            failed_tests.append(label)

        if failed_once:
            failed_at_least_once_names.append(label)

    return totals, per_suite, passed_tests, failed_tests, flaky_tests, failed_at_least_once_names

# ---- Aggregate intents (extended) ----
def handle_aggregate_intents(message: str) -> Optional[str]:
    msg = message.lower().strip()
    totals, per_suite, passed_tests, failed_tests, flaky_tests, failed_once_names = get_snapshot()

    # Any flaky?
    if re.search(r"\b(any\s+)?flaky tests?\b|\bflaky\?\b", msg):
        if totals.get("flaky", 0) == 0:
            return "No flaky tests."
        items = "\n".join(f"- {t}" for t in flaky_tests[:50])
        more = "" if len(flaky_tests) <= 50 else f"\n… and {len(flaky_tests)-50} more."
        return f"{totals['flaky']} flaky tests:\n{items}{more}"

    # Counts (passed/failed/skipped/flaky)
    m = re.search(r"(how many|count|number of)\s+(tests?\s+)?(passed|failed|skipped|flaky)\b", msg)
    if m:
        what = m.group(3)
        return f"{totals.get(what,0)} tests {what}."

    # Total tests
    if re.search(r"(how many|count|number of)\s+(tests?)\b", msg):
        total = sum(totals.values())
        return f"Total tests: {total} (passed {totals['passed']}, failed {totals['failed']}, skipped {totals['skipped']}, flaky {totals['flaky']})."

    # List FINAL failed tests
    if re.search(r"\b(list|show|which)\b.*\b(failed tests|tests that failed|failures)\b", msg):
        if not failed_tests:
            return "No failed tests (final results)."
        items = "\n".join(f"- {t}" for t in failed_tests[:50])
        more = "" if len(failed_tests) <= 50 else f"\n… and {len(failed_tests)-50} more."
        return f"Failed tests (final):\n{items}{more}"

    # List tests that failed at least once (even if they passed finally)
    if re.search(r"\b(list|show|which)\b.*\b(failed at least once|failed on retry|failed during retry|failed once)\b", msg):
        recovered = failed_once_names
        if not recovered:
            return "No tests failed during retries; all passing tests stayed green."
        items = "\n".join(f"- {t}" for t in recovered[:100])
        more = "" if len(recovered) <= 100 else f"\n… and {len(recovered)-100} more."
        return f"Tests that failed at least once:\n{items}{more}"

    # List passed tests
    if re.search(r"\b(list|show|which)\b.*\b(passed tests|tests that passed)\b", msg):
        if not passed_tests:
            return "No passed tests."
        items = "\n".join(f"- {t}" for t in passed_tests[:50])
        more = "" if len(passed_tests) <= 50 else f"\n… and {len(passed_tests)-50} more."
        return f"Passed tests:\n{items}{more}"

    # Counts by suite
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

    # 0) Greetings first
    if is_greeting(text):
        updated = _append_exchange(history_msgs, user_text, GREETING_REPLY)
        return updated, gr.update(value="")

    def _short_help() -> str:
        totals, *_ = get_snapshot()
        total = sum(totals.values())
        lines = [
            "Here’s the current snapshot:",
            f"• Total: {total}",
            f"• Passed: {totals['passed']}",
            f"• Failed (final): {totals['failed']}",
            f"• Skipped: {totals['skipped']}",
            f"• Flaky: {totals['flaky']}",
            "",
            "Try: 'list failed tests', 'list tests that failed at least once', 'any flaky tests?', or 'how many passed in <suite>'.",
        ]
        return "\n".join(lines)

    # 1) Too vague → quick snapshot
    if re.fullmatch(r"(?i)(tests?|results?)\??", text):
        updated = _append_exchange(history_msgs, user_text, _short_help())
        return updated, gr.update(value="")

    # 2) Aggregate intents
    try:
        agg = handle_aggregate_intents(text)
        if agg is not None:
            updated = _append_exchange(history_msgs, user_text, agg)
            return updated, gr.update(value="")
    except Exception:
        pass  # keep chat alive

    # 3) Fallback: NB model
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
    msg = gr.Textbox(placeholder="Say hi or ask: How many tests passed?")
    send = gr.Button("Send", variant="primary")

    send.click(fn=respond, inputs=[msg, chat], outputs=[chat, msg])
    msg.submit(fn=respond, inputs=[msg, chat], outputs=[chat, msg])
    refresh_btn.click(fn=do_refresh, outputs=refresh_status)

if __name__ == "__main__":
    demo.launch()

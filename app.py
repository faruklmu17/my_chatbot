# app.py
import os
import re
import joblib
import traceback
from typing import Optional, Dict

from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

# Your model artifacts
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
ANSWERS_PATH = "answers.pkl"

# Timezone for human-friendly timestamps
LOCAL_TZ = ZoneInfo("America/New_York")

# Import your data helpers
import train_model  # uses fetch_data(), iter_tests_with_time(), iter_tests()

# Auto-enable a tiny Gradio UI on Hugging Face Spaces (keeps local behavior unchanged)
ENABLE_MINIMAL_UI = bool(os.getenv("SPACE_ID")) or os.getenv("ENABLE_MINIMAL_UI") == "1"

# ---------------- Model/answers ----------------

model = None
vectorizer = None
answers: Dict[str, str] = {}

def load_artifacts():
    """Load NB model, vectorizer, and canned answers."""
    global model, vectorizer, answers
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    answers = joblib.load(ANSWERS_PATH)

# ---------------- Helpers: last-run + greetings ----------------

def get_last_run_info() -> str:
    """Find the most recent test attempt start time and format it nicely."""
    data = train_model.fetch_data()
    latest = None
    sample_name = None
    sample_project = None

    for _suite, _spec, test_title, _status, project, _flaky, start_dt in train_model.iter_tests_with_time(data):
        if not start_dt:
            continue
        if (latest is None) or (start_dt > latest):
            latest = start_dt
            sample_name = test_title
            sample_project = project

    if not latest:
        return "I couldn't find a timestamp for the last run."

    local_dt = latest.astimezone(LOCAL_TZ)
    # latest is tz-aware UTC datetime from iter_tests_with_time
    return (
        f"Last test attempt started on {local_dt.strftime('%Y-%m-%d %I:%M:%S %p %Z')} "
        f"(~{latest.strftime('%Y-%m-%d %H:%M:%S')} UTC)"
        + (f" — example test: “{sample_name}”" if sample_name else "")
        + (f" (project: {sample_project})" if sample_project else "")
        + "."
    )

def is_greeting(text: str) -> bool:
    """Lightweight greeting detector (hi/hello/gm/hey/how are you, etc.)."""
    patterns = [
        r"^\s*hi\b",
        r"^\s*hello\b",
        r"^\s*hey\b",
        r"\bgood\s*(morning|afternoon|evening)\b",
        r"\bhow\s*are\s*you\b",
        r"\bhow\s*r\s*you\b",
        r"\bgm\b", r"\bgn\b", r"\bge\b",
    ]
    t = (text or "").strip().lower()
    return any(re.search(p, t, re.I) for p in patterns)

# ---------------- Intent classification + response ----------------

def respond(user_text: str) -> str:
    """
    Core responder. Keep your UI exactly as-is and just call respond(...) as before.
    Adds:
      - greeting replies
      - 'last run' intent (works via regex even if you don't retrain)
      - still supports your NB canned answers
    """
    try:
        txt = (user_text or "").strip()

        # 1) Greetings
        if is_greeting(txt):
            return ("Hi! 👋 I’m your Playwright test helper. "
                    "You can ask things like: 'list failed tests', 'how many passed', "
                    "'last run test', or 'why did a test fail?'")

        # 2) Try your Naive Bayes model first
        key: Optional[str] = None
        try:
            vec = vectorizer.transform([txt])
            pred = model.predict(vec)[0]
            key = pred if isinstance(pred, str) else None
        except Exception:
            key = None

        # 3) Regex fallback for last-run (so you don't *need* to retrain)
        last_run_match = any(re.search(p, txt, re.I) for p in [
            r"\blast\s+(test\s*)?run\b",
            r"\bmost\s+recent\s+test\b",
            r"\bwhen\s+did\s+the\s+last\s+test\s+run\b",
            r"\blast\s+run\s+test\b",
            r"\bprevious\s+run\b",
        ])
        if last_run_match or key == "last_run":
            return get_last_run_info()

        # 4) Use your canned answers (answers.pkl)
        if key and key in answers:
            val = answers[key]
            if isinstance(val, str) and val.strip().upper() == "LAST_RUN_PLACEHOLDER":
                return get_last_run_info()
            return val

        # 5) Default fallback
        return ("I didn't catch that. Try: 'list failed tests', 'how many passed', "
                "'last run test', 'test duration <name>', or 'why did the test fail?'.")
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}"

# ---------------- Minimal optional UI (auto-enabled on HF Spaces) ----------------

if __name__ == "__main__":
    load_artifacts()

    if ENABLE_MINIMAL_UI:
        import gradio as gr

        with gr.Blocks(title="Playwright Test Q&A") as demo:
            gr.Markdown("## Playwright Test Q&A")
            chat_in = gr.Textbox(label="Ask me about the tests")
            chat_out = gr.Textbox(label="Answer")
            btn = gr.Button("Ask")

            def _on_click(q):
                return respond(q)

            btn.click(_on_click, inputs=[chat_in], outputs=[chat_out])
            chat_in.submit(_on_click, inputs=[chat_in], outputs=[chat_out])

        # On HF, Gradio will bind to the port it detects (SPACE settings). Locally, default port.
        demo.launch()
    else:
        # No UI here — this keeps your existing UI untouched (import respond() elsewhere).
        print("Artifacts loaded. Using existing UI. (Set ENABLE_MINIMAL_UI=1 or run on HF Spaces to launch demo.)")

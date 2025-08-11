# app.py
import re
import gradio as gr
import joblib
import traceback
from typing import Optional, Dict, Any

from datetime import datetime, timezone
from zoneinfo import ZoneInfo  # Python 3.9+, converts UTC -> local tz

import train_model  # reuse fetch_data(), iter_tests(), iter_tests_with_time()

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
ANSWERS_PATH = "answers.pkl"

# Set your local timezone (user is in New York)
LOCAL_TZ = ZoneInfo("America/New_York")

model = None
vectorizer = None
answers: Dict[str, str] = {}

def load_artifacts():
    global model, vectorizer, answers
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    answers = joblib.load(ANSWERS_PATH)

# ---------- NEW: last-run extractor ----------

def get_last_run_info() -> str:
    """
    Scan the JSON and return a friendly summary of the most recent test attempt start time.
    """
    data = train_model.fetch_data()
    latest_utc: Optional[datetime] = None
    latest_test_name = None
    latest_project = None

    for suite_title, spec_title, test_title, final_status, project, is_flaky, start_dt in train_model.iter_tests_with_time(data):
        if not start_dt:
            continue
        if (latest_utc is None) or (start_dt > latest_utc):
            latest_utc = start_dt
            latest_test_name = test_title
            latest_project = project

    if not latest_utc:
        return "I couldn't find a timestamp for the last run."

    local_dt = latest_utc.astimezone(LOCAL_TZ)
    return (
        f"Last test attempt started on {local_dt.strftime('%Y-%m-%d %I:%M:%S %p %Z')} "
        f"(~{latest_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC)"
        + (f" — example test: “{latest_test_name}”" if latest_test_name else "")
        + (f" (project: {latest_project})" if latest_project else "")
        + "."
    )

# ---------- Existing helper (if you have one) ----------
def classify_intent(user_text: str) -> Optional[str]:
    """
    Use the trained model to map to an answer key; fall back to pattern match
    for 'last run' queries so you don't need to retrain immediately.
    """
    try:
        vec = vectorizer.transform([user_text])
        pred_idx = model.predict(vec)[0]
        # If your model predicts an index and you index into answers differently, adjust as needed.
        # Assuming model predicts an index into a list of keys or we stored mapping in answers:
        # If you stored keys directly, you may already get a string key.
        key = pred_idx if isinstance(pred_idx, str) else None
    except Exception:
        key = None

    # --- Fallback keyword/regex for "last run" intent ---
    last_run_patterns = [
        r"\blast\s+(test\s*)?run\b",
        r"\bmost\s+recent\s+test\b",
        r"\bwhen\s+did\s+the\s+last\s+test\s+run\b",
        r"\blast\s+run\s+test\b",
        r"\bprevious\s+run\b",
    ]
    if any(re.search(p, user_text, flags=re.I) for p in last_run_patterns):
        return "last_run"

    return key

def respond(user_text: str) -> str:
    try:
        key = classify_intent(user_text)

        # If your training data already includes a "last_run" key in answers.pkl,
        # you can keep that mapping; we still answer dynamically here:
        if key == "last_run":
            return get_last_run_info()

        # Otherwise, use the usual FAQ/intents from your Naive Bayes:
        if key and key in answers:
            # If some answers are placeholders like "LAST_RUN_PLACEHOLDER", handle here:
            if answers[key].strip().upper() == "LAST_RUN_PLACEHOLDER":
                return get_last_run_info()
            return answers[key]

        # If nothing matched, you can provide a safe default or echo:
        return "I didn't catch that. Try asking things like: 'list failed tests', 'last run test', or 'how many passed?'"

    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}"

# ---------- Gradio UI (minimal) ----------
with gr.Blocks(title="Playwright Test Bot") as demo:
    gr.Markdown("## Playwright Test Q&A")
    chat_in = gr.Textbox(label="Ask me about the tests")
    chat_out = gr.Textbox(label="Answer")
    btn = gr.Button("Ask")

    def _on_click(q):
        return respond(q)

    btn.click(_on_click, inputs=[chat_in], outputs=[chat_out])
    chat_in.submit(_on_click, inputs=[chat_in], outputs=[chat_out])

if __name__ == "__main__":
    load_artifacts()
    demo.launch()

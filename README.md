---
title: Playwright Chatbot
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
tags:
  - testing
  - playwright
  - qa
---
# 🤖 Playwright Test Bot – JSON Chatbot with Gradio

This is a simple, lightweight **chatbot** built with **Python**, **Naive Bayes**, and **Gradio**, designed to answer questions about Playwright test results stored in a **JSON file**.

You can ask questions like:
- "What was the result of 'test valid signup'?"
- "Did the login test pass?"
- "How many tests failed?"

---

## 📦 Features

- ✅ Automatically fetches a **Playwright test result JSON** file from a GitHub repo
- ✅ Dynamically **trains a Naive Bayes model** on test data (via separate script)
- ✅ Provides a **Gradio web UI** to chat with the test results
- ✅ Re-trains itself when the test result file changes
- ✅ Fully **offline-compatible**, no LLMs or OpenAI API needed

---

## 🚀 Getting Started (Step-by-Step)

1. **Clone the Repo**
   ```bash
   git clone https://github.com/your-username/playwright-bot.git
   cd playwright-bot
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**
   Run this to fetch the latest JSON from GitHub and train the model:
   ```bash
   python train_model.py
   ```

4. **Run the Chatbot**
   Launch the Gradio chatbot UI:
   ```bash
   python app.py
   ```

---

## 🔍 How It Works

### 1. **Loads JSON from GitHub**

The chatbot loads the test result JSON from a GitHub repo using the `raw.githubusercontent.com` link.

### 2. **Generates Q&A Pairs**

Each test case in the JSON is turned into question-answer pairs like:
- **Q**: What was the result of `test valid signup`?
- **A**: The test `test valid signup` in `tests/test_signup.spec.js` passed.

### 3. **Trains a Naive Bayes Model**

`train_model.py` trains the model and saves:
- `model.pkl` (trained model)
- `vectorizer.pkl` (text vectorizer)
- `answers.pkl` (answer list)

### 4. **Gradio UI for Interaction**

`app.py` loads the saved model and answers to create a chatbot interface.

---

## 🔐 Working with Public vs Private Repos

### ✅ Public GitHub Repo

If your Playwright test result JSON is in a **public repo**, you can directly use the `raw.githubusercontent.com/...` URL like this:

```
https://raw.githubusercontent.com/username/repo/main/tests/test-results.json
```

No authentication needed.

---

### 🔐 Private GitHub Repo

GitHub will block access to the raw file unless authorized. You have **3 options**:

#### 🔒 Option 1: Use a Personal Access Token (PAT)
Use Python’s `requests` with headers:
```python
headers = {
    "Authorization": "token YOUR_PAT"
}
```

#### 💻 Option 2: Clone Locally
Clone your private repo locally and point to the file directly:
```python
with open("tests/test-results.json") as f:
    data = json.load(f)
```

#### 📡 Option 3: Use GitHub API
Use the GitHub Contents API with authentication and decode the base64 content.

---

## 🧱 Project Structure

```
.
├── app.py                # Gradio chatbot interface
├── train_model.py        # Loads JSON & trains model
├── model.pkl             # Trained model (generated)
├── vectorizer.pkl        # Trained vectorizer (generated)
├── answers.pkl           # Saved answer list (generated)
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

---

## 🛠 Dependencies

Install everything in one line with:
```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Author

Created by [**Faruk Hasan**](https://faruk-hasan.com/)  
Senior QA Engineer | Coding Instructor | AI Explorer

---

## 📄 License

MIT License
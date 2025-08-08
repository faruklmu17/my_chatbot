# 🤖 Playwright Test Bot – JSON Chatbot with Gradio

This is a simple, lightweight **chatbot** built with **Python**, **Naive Bayes**, and **Gradio**, designed to answer questions about Playwright test results stored in a **JSON file**.

You can ask questions like:
- "What was the result of 'test valid signup'?"
- "Did the login test pass?"
- "How many tests failed?"

---

## 📦 Features

- ✅ Automatically fetches a **Playwright test result JSON** file from a GitHub repo
- ✅ Dynamically **trains a Naive Bayes model** on test data
- ✅ Provides a **Gradio web UI** to chat with the test results
- ✅ Re-trains itself when the test result file changes
- ✅ Fully **offline-compatible**, no LLMs or OpenAI API needed

---

## 🔍 How It Works

### 1. **Loads JSON from GitHub**

The chatbot loads the test result JSON from a GitHub repo using the `raw.githubusercontent.com` link.

### 2. **Generates Q&A Pairs**

Each test case in the JSON is turned into question-answer pairs like:
- **Q**: What was the result of `test valid signup`?
- **A**: The test `test valid signup` in `tests/test_signup.spec.js` passed.

### 3. **Trains a Naive Bayes Model**

The model is trained on these Q&A pairs using `CountVectorizer` + `MultinomialNB` from `scikit-learn`.

### 4. **Gradio UI for Interaction**

You can ask natural questions in a browser using the Gradio interface.

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

## 🚀 Hosting on Hugging Face Spaces

This chatbot can be deployed for free using **[Hugging Face Spaces](https://huggingface.co/spaces)**.

1. Create a new **Gradio Space**
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
3. Done!

---

## 📁 File Structure

```
.
├── app.py                # Main Gradio chatbot app
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🛠 Dependencies

```bash
pip install gradio scikit-learn requests
```

---

## 👨‍💻 Author

Created by **Faruk Hasan**  
Senior QA Engineer | Coding Instructor | AI Explorer

---

## 📄 License

MIT License
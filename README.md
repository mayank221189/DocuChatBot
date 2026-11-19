# 📄 DocuChatBot

A Streamlit app for asking questions about an uploaded PDF. It extracts PDF text, creates an in-memory vector index, retrieves relevant passages, and uses Google Gemini to answer questions.

## ✨ Features

- Upload a PDF and chat with its contents.
- Uses `pypdf` for text extraction and LangChain for chunking and retrieval.
- Keeps the vector index and chat history in the Streamlit session.
- Replacing the upload removes existing files in `documents/` and saves the new PDF there under its original filename.

## 🧰 Requirements

- Python 3.12 or newer
- A Google Gemini API key with access to the models configured in `main.py`
- Internet access for Google Gemini embeddings and chat responses

## 🚀 Install

### With uv

```bash
uv sync
```

### With pip

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## 🔑 Configure the API key

Create a `.env` file in the project root:

```env
GOOGLE_GEMINI_API_KEY=your_google_gemini_api_key
```

The `.env` file is ignored by Git. Keep your API key private.

## ▶️ Start the app

With uv:

```bash
uv run streamlit run main.py
```

With the activated pip environment:

```bash
streamlit run main.py
```

Streamlit prints a local URL in the terminal; open it in your browser, upload a PDF, and enter a question in the chat.

## ℹ️ Notes

- PDF text extraction uses `pypdf`; image-only or scanned PDFs may not produce usable text.
- The vector index is kept in memory, so upload and index the PDF again after restarting the app or losing the Streamlit session.
- PDF text is sent to Google for embeddings, and retrieved text is sent to Gemini for answers. Only use sensitive documents if this is permitted by your data-handling requirements.
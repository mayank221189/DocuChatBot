# 📄 DocuChatBot

A local-first Streamlit app for asking questions about an uploaded PDF. It extracts text, builds an in-memory vector index, and uses Ollama for local embeddings and chat responses.

## ✨ Features

- Upload a PDF and chat with its contents.
- Uses `pypdf` for text extraction and LangChain for chunking and retrieval.
- Runs both the embedding model and chat model through Ollama on `localhost`.
- Processes PDF bytes in memory and keeps the vector index and chat history in the Streamlit session.

## 🧰 Requirements

- Python 3.12 or newer
- [Ollama](https://ollama.com/download)
- Enough memory for the selected local models

## 🚀 Setup

### 1. Install and start Ollama

On macOS with Homebrew:

```bash
brew install ollama
brew services start ollama
```

Or install Ollama from its [official download page](https://ollama.com/download) and start the Ollama app.

### 2. Download the models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

These are the chat and embedding models configured in `main.py`. The model files are downloaded once; PDF text and prompts are sent only to the local Ollama service.

### 3. Install the Python packages

With uv:

```bash
uv sync
```

Or with pip:

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

## ▶️ Run the app

Make sure Ollama is running, then start Streamlit:

```bash
uv run streamlit run main.py
```

With the activated pip environment, use:

```bash
streamlit run main.py
```

Open the local URL printed by Streamlit, upload a PDF, and ask a question.

## ℹ️ Notes

- PDF extraction uses `pypdf`; image-only or scanned PDFs may not produce usable text.
- The PDF and vector index are kept in memory. Upload and index the PDF again after restarting the app or losing the Streamlit session.
- The app connects only to Ollama at `http://localhost:11434`; it does not send document text or prompts to Google.
- This configuration expects Ollama to run on the same host as Streamlit, so it is not suitable for Streamlit Community Cloud without a separately hosted Ollama service.
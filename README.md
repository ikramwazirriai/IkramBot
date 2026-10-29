# Ikrambot

Ikrambot is a Streamlit chat app that uses retrieval-augmented generation (RAG) to answer questions about Ikram Ullah. It retrieves supporting passages from the included knowledge base and uses Google Gemini to write an answer.

## Run in VS Code

1. Install Python 3.11 and the VS Code Python extension.
2. Open the `IkramBot` folder in VS Code (the folder containing `app.py`).
3. In the VS Code terminal, create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   On macOS/Linux, use `python3.11 -m venv .venv` and `source .venv/bin/activate`.
4. Install dependencies: `python -m pip install -r requirements.txt`.
5. Create a `.env` file in this folder with your Gemini API key:

   ```text
   GEMINI_API_KEY=your_key_here
   ```

6. Start the app: `streamlit run app.py`.

The app opens at `http://localhost:8501`. Never commit `.env` or paste an API key into source code.

## Deploy on Streamlit Community Cloud

1. Push this repository to a GitHub account you can access.
2. In Streamlit Community Cloud, create an app and select this repository, the `main` branch, and `app.py` as the main file.
3. In the app's **Settings → Secrets**, add:

   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```

4. Deploy. The FAISS index and metadata are included in the repository, so the app does not need to rebuild embeddings at startup. The first launch downloads the `all-MiniLM-L6-v2` embedding model and may take a few minutes.

## Knowledge base

PDF sources are stored in `data/`; the prebuilt FAISS index and corresponding metadata are in `vectorstore/`. To rebuild the index after changing the PDFs, run:

```bash
python src/build_vectorstore.py
```

Rebuilding requires the embedding model to be downloaded once. Keep the PDFs, index, and metadata in sync.

## Configuration

- Local development: set `GEMINI_API_KEY` in a root `.env` file.
- Streamlit Community Cloud: set `GEMINI_API_KEY` in app Secrets.
- Dependencies: `requirements.txt`.
- VS Code: `.vscode/launch.json` provides a Streamlit debug configuration.

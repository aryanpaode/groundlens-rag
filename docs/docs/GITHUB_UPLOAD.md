# Publish this repository

**Recommended repository name:** `groundlens-rag`

**Owner:** use your `aryanpaode` account.

**Description:**

> Evidence-grounded multi-PDF question answering using local embeddings, ChromaDB and Gemini, with page-level citations, a Streamlit interface and recorded evaluation results.

**Suggested topics:** `rag`, `retrieval-augmented-generation`, `document-question-answering`, `pdf`, `streamlit`, `chromadb`, `gemini`, `sentence-transformers`.

## Browser upload

1. Extract the ZIP to a separate folder. Leave the already-working GroundLens application untouched.
2. On GitHub, create a repository named `groundlens-rag`. Choose **Public** for a publicly viewable portfolio repository, or Private where required by your course. Do not initialize a second README, `.gitignore` or licence: this package already supplies the README and ignore rules, and no licence has been selected.
3. Open **uploading an existing file** on the empty-repository page, or **Add file → Upload files** in an initialized repository. Open the extracted `groundlens-rag` folder on your computer and drag its **contents** into the upload area. Upload the actual files and subfolders, not the ZIP and not an extra enclosing folder.
4. Confirm that `README.md`, `app.py`, `.env.example`, `.gitignore`, `.streamlit/config.toml`, `rag/`, `docs/` and the other folders are included. The README and app must be at the repository root. Commit with the message `Add GroundLens final RAG application, report and evaluation evidence`.

After uploading, open the README and check that its dashboard image, screenshot links and report link work. This is source publication, not deployment of a public Streamlit application. No public repository or deployment has been created by preparing this ZIP.

## Alternative: Git from the extracted clean folder

Use these commands only for a new local folder and an empty remote repository created under the stated owner/name. Authentication must already be configured or completed when Git asks.

```powershell
git init
git add .
git status
git commit -m "Add GroundLens final RAG application, report and evaluation evidence"
git branch -M main
git remote add origin https://github.com/aryanpaode/groundlens-rag.git
git push -u origin main
```

Inspect `git status` before committing. Actual `.env`, `runtime/` and `.venv/` must not appear. Do not blindly run remote-creation commands inside an existing repository with another configured remote.

## Final report

`docs/report/GroundLens_Final_Report_NMIMS_Refined.docx` is the finalized report supplied earlier. Its publication-status section records that no live repository URL had been supplied at that time. After successfully publishing, the confirmed address can be inserted into the submission copy; this distribution does not claim that an uncreated URL is live.

## Official GitHub instructions

- Creating a repository: https://docs.github.com/en/get-started/start-your-journey/creating-a-repository-for-your-project-on-github
- Uploading files and folders: https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
- Adding local code with Git: https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github

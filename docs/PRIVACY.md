# Privacy and publication boundaries

## Data path

PDF text extraction, sentence embeddings and vector search run locally. The app stores extracted text and metadata in `runtime/chroma/`; this folder can contain document content and must not be published. The indexing code does not retain the original uploaded PDF file.

Gemini receives the current question, selected text excerpts and limited recent conversation context. Local indexing does not make the application fully offline or prevent the selected excerpts from leaving the machine. Use public, synthetic or appropriately authorized material.

## Credentials

Copy `.env.example` to `.env` and insert your own key only on your machine. The distribution contains no working API key. Do not share `.env`, screenshots of secrets, private credential files or `.streamlit/secrets.toml`.

The app masks its configured key in displayed exception text, but this is not a general-purpose secret-redaction system or a security audit. Review material before making it public.

## What this distribution excludes

The package excludes actual credentials, virtual environments, runtime databases, model weights, complete third-party lecture PDFs and obsolete draft reports. It includes synthetic demonstration PDFs and the final project report with the author's original application captures. Those captures show limited lecture excerpts as evaluation evidence; they do not grant rights to redistribute full source documents.

The NMIMS/MPSTME identification in the author's report and the UI colour theme do not imply that the software is an official institutional product. Third-party material is not relicensed by inclusion in a screenshot or report. No repository-wide licence has been selected in this distribution.

## Upload from a clean copy

Use the supplied clean repository folder for publication, not the working application folder that contains your key and index. Git ignore rules help when committing with Git, but are not a substitute for inspecting the files selected in a browser upload.

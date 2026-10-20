# Data handling and publication

GroundLens combines local processing with external services. PyMuPDF extraction and MiniLM embeddings run on the laptop. Pinecone receives chunk vectors, extracted text and source metadata. Gemini receives the question, retrieved excerpts and limited recent conversation context. Use only documents permitted for those services.

## Private configuration

Keep Pinecone and Gemini keys only in `.env` or an appropriate private secret store. `.env.example` contains placeholders. Do not share `.env`, `.venv`, model caches, `runtime/`, API credentials or full restricted lecture PDFs.

The supplied `.gitignore` is a repository aid, not a substitute for inspecting files before a browser upload. This package physically excludes private runtime data and credentials.

## Stored records

Preserve the working namespace and `runtime/pinecone_manifest.json`. The namespace identifies remote records; the local manifest tracks this installation's documents and counts. Erasing the local manifest does not delete Pinecone vectors. A new local installation starts with an empty manifest and must index its own permitted documents; it does not automatically reconstruct another installation's inventory from Pinecone.

Document removal deletes the exact document records. Clearing all indexed documents deletes the configured namespace's vectors, not the whole Pinecone index. Do not share that namespace with unrelated applications. These are explicit data-deletion operations, unrelated to updating the source repository.

## Included material

The repository includes the approved report, original application screenshots, prepared evaluation questions, recorded measurements and three clearly labelled fictional sample PDFs. Full institutional lecture PDFs, original scans, private notes and generated vector records are excluded. Screenshots still contain selected lecture excerpts and should be published only when permitted. No blanket redistribution licence is granted over institutional logos or third-party material.


## Optional advanced stages

Docling and semantic embeddings run locally. Contextualization, HyDE and hierarchical summarization can send additional document-derived text to Gemini. DSPy optimization sends examples to the configured provider. RAGAS sends real answer/context records to its judge. Langfuse receives question/answer and observed metadata only after explicit opt-in.

Use `scripts/build_repo_zip.py` to create a credential-free publication snapshot from a working folder. Keep `.local_backups`, all `.venv*` folders, runtime data, saved DSPy JSON and live CSV exports private. The archive builder uses known-file allowlists and basic secret-pattern checks, not an exhaustive security audit of screenshot or document content.

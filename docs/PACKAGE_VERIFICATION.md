# Publication package verification

## Source and report provenance

The package combines the supplied Pinecone/LangGraph implementation, the last supplied burgundy/red/white UI patch, the fifteen original final-system screenshots and the approved final report. It is prepared for the existing `groundlens-rag` repository; it is not a new application implementation.

The retrieval and generation behavior is retained. The publication edits remove release-label prose, standardize the academic benchmark filename and screenshot names, refresh README links and help text, and omit obsolete material. The configured embedding checkpoint, Pinecone index and namespace defaults are not changed, because they identify the working resources.

The report is **byte-for-byte identical** to the approved submission file.

Report SHA-256: `f3dc228a39e7076741cca57c301b556cd605f70328aafc6d0c88ae0662a59c6f`

## Packaging checks executed

- Python syntax/compile checks passed for the application, modules and tests.
- The available offline suite returned **21 passed, 1 skipped in 0.36 seconds** on Python 3.13.5 in the packaging environment.
- The skipped item is the LangGraph integration-test module because `langgraph` is not installed in this environment. It contains two test functions. The separately supplied Windows setup record reports **23 passed in 1.66 seconds** with the required libraries installed.
- The Python 3.11 Windows application runs and live question answers are evidenced by the student's records, not a new cloud session during packaging.
- No Pinecone/Gemini credentials were available here. No live requests, new indexing, complete Streamlit browser run or provider-availability test was performed during packaging.
- All fifteen published screenshots retain the exact bytes of their source captures. Their hashes are listed in `screenshots/SHA256SUMS.txt`.
- The relative Markdown links and image paths were checked locally. A relative-link check is not a live GitHub publication check.
- Private `.env` files, virtual environments, runtime manifests, stored vectors, caches, model weights and full lecture PDFs are physically excluded.

## Publication boundary

This package does not modify the live GitHub repository or the user's working installation. Live repository access was unavailable during packaging, so branch contents must be checked before merge. Updating the GitHub source does not erase Pinecone data or rewrite commit history.

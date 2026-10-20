# Release hand-off

The consolidated application and the planned integrations have recorded functional demonstrations. RAGAS and Langfuse are no longer marked as awaiting their first live run: the successful screenshots are included. See RESULTS.md for the exact scope of each result.

This release packages the established application, the 240-second RAGAS timeout, a separate compatible evaluation requirements file, the updated report and evidence. It does not add new product features or rewrite your Pinecone vectors.

The private laptop state is not available in this archive. Use `scripts/restore_local_state.py` from a fresh extraction before discarding an older installation. Add `.env` manually. The source installation and cloud collection remain unchanged.

For repository upload, use the clean supplied source or `scripts/build_repo_zip.py`, not the entire working directory after adding keys. Existing history and the repository URL can remain intact while tracked files are updated. Do not publish institutional source PDFs without permission.

No broad controlled accuracy benchmark or statistical improvement claim is made. The RAGAS value of 1.0000 belongs to one answer; the report states that boundary.

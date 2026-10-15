"""Pinecone configuration check; may create a missing index, but uploads no documents and prints no keys."""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / '.env')
key=os.getenv('PINECONE_API_KEY','').strip()
if not key or key.startswith('REPLACE_'):
    raise SystemExit('PINECONE_API_KEY missing. Configure your private .env first.')
from rag.pinecone_store import PineconeVectorStore

try:
    store=PineconeVectorStore(
        api_key=key,
        index_name=os.getenv('PINECONE_INDEX','groundlens-rag-384'),
        namespace=os.getenv('PINECONE_NAMESPACE','groundlens-v3'),
        manifest_file=Path(__file__).resolve().parent / 'runtime' / 'pinecone_manifest.json',
        cloud=os.getenv('PINECONE_CLOUD','aws'),
        region=os.getenv('PINECONE_REGION','us-east-1'))
except Exception as exc:
    msg=str(exc).replace(key,'[REDACTED]')
    raise SystemExit(f'Pinecone not ready: {type(exc).__name__}: {msg}') from exc
print('Pinecone connected successfully.')
print('Index:',store.index_name)
print('Namespace:',store.namespace)
print('Locally tracked files:',len(store.list_documents()))
print('Locally tracked vectors:',store.count)
print('Connection check complete. This command does not upload document vectors.')

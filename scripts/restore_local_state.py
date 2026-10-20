"""Restore private GroundLens state into a clean extraction without re-indexing.

Uses only the Python standard library. Does not copy .env or virtual environments,
call external services, or change the source installation. Stop both applications
before running so that the manifest and the database describe the same state.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_FILES = (
    'runtime/pinecone_manifest.json',
    'runtime/groundlens_bm25.sqlite3',
    'artifacts/dspy_program.json',
    'evaluation/groundlens_ragas_input.csv',
    'evaluation/ragas_ready.csv',
    'evaluation/ragas_results.csv',
)


def _corpus_settings(path: Path) -> dict[str, str]:
    """Read only corpus identifiers, never return keys or other private values."""
    found: dict[str, str] = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        match = re.match(r'^\s*(?:export\s+)?(PINECONE_INDEX|PINECONE_NAMESPACE)\s*=\s*(.*?)\s*$', line)
        if match:
            value = match[2]
            if value.startswith(('"', "'")):
                end = value.find(value[0], 1)
                if end < 1:
                    raise ValueError(f'Unclosed quote in a corpus setting in {path.name}')
                value = value[1:end]
            else:
                value = re.split(r'\s+#', value, maxsplit=1)[0].strip()
            found[match[1]] = value
    return found


def _check_settings(env_file: Path, manifest: dict) -> None:
    settings = _corpus_settings(env_file)
    for name, field in [('PINECONE_INDEX', 'index'), ('PINECONE_NAMESPACE', 'namespace')]:
        if name in settings and settings[name] != manifest[field]:
            raise ValueError(f'{env_file}: {name} does not match the source manifest. No state copied.')


def _snapshot_sqlite(source: Path, destination: Path) -> None:
    """SQLite's backup API includes committed WAL data without raw WAL copying."""
    deadline = time.monotonic() + 60
    def progress(status: int, remaining: int, total: int) -> None:
        if time.monotonic() > deadline:
            raise TimeoutError('Database backup exceeded 60 seconds. Stop GroundLens and retry.')
    src = sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True, timeout=10)
    dst = sqlite3.connect(destination)
    try:
        src.backup(dst, pages=256, progress=progress, sleep=0.1)
        if dst.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('The copied BM25 database failed its integrity check.')
    finally:
        dst.close()
        src.close()


def restore(source: Path, destination: Path, *, overwrite: bool = False) -> dict:
    source, destination = source.expanduser().resolve(), destination.expanduser().resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Use separate, non-nested source and destination project folders.')
    if not (destination / 'app.py').is_file():
        raise FileNotFoundError('Destination must be the extracted GroundLens folder containing app.py.')
    manifest_path = source / STATE_FILES[0]
    if not manifest_path.is_file():
        raise FileNotFoundError(f'No source manifest at {manifest_path}. Choose the current working project.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if (not isinstance(manifest, dict) or not isinstance(manifest.get('documents'), dict)
            or not isinstance(manifest.get('index'), str) or not isinstance(manifest.get('namespace'), str)):
        raise ValueError('Source manifest structure is invalid. No state copied.')
    _check_settings(source / '.env', manifest)
    _check_settings(destination / '.env', manifest)
    existing_manifest = destination / STATE_FILES[0]
    if existing_manifest.is_file():
        current = json.loads(existing_manifest.read_text(encoding='utf-8-sig'))
        if any(current.get(k) != manifest[k] for k in ('index', 'namespace')):
            raise ValueError('Destination holds another collection. Use a fresh extraction.')
    available = [name for name in STATE_FILES if (source / name).is_file()]
    missing = [name for name in STATE_FILES if name not in available]
    collisions = [name for name in STATE_FILES if (destination / name).exists()]
    # Never mix a stale local DSPy program/database with newly copied state.
    if collisions and not overwrite:
        raise FileExistsError('Destination already has private state. Use a fresh extraction, or --overwrite for a backed-up replacement.')
    if any((destination / (STATE_FILES[1] + suffix)).exists() for suffix in ('-wal', '-shm')):
        raise ValueError('Destination has SQLite sidecar files. Stop its app and use a fresh extraction.')
    for name in ('artifacts/dspy_program.json',):
        if name in available:
            payload = json.loads((source/name).read_text(encoding='utf-8-sig'))
            if not isinstance(payload, dict) or not payload:
                raise ValueError('Source DSPy program is not a nonempty JSON object. No state copied.')

    backup_dir = None
    with tempfile.TemporaryDirectory(prefix='.state_import_', dir=destination) as temp:
        stage = Path(temp)
        for name in available:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith('.sqlite3'):
                _snapshot_sqlite(source/name, target)
            else:
                shutil.copy2(source/name, target)
        if collisions:
            backup_dir = destination / '.local_backups' / (datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
            for name in collisions:
                backup = backup_dir/name
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination/name, backup)
        changed: list[str] = []
        try:
            for name in STATE_FILES:
                target = destination/name
                if name in available:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    (stage/name).replace(target)
                    changed.append(name)
                elif name in collisions:
                    target.unlink()  # old file already preserved in backup
                    changed.append(name)
        except Exception:
            for name in reversed(changed):
                target = destination/name
                if backup_dir is not None and (backup_dir/name).exists():
                    shutil.copy2(backup_dir/name, target)
                else:
                    target.unlink(missing_ok=True)
            raise
    return {'copied': available, 'missing': missing, 'index': manifest['index'],
            'namespace': manifest['namespace'], 'documents': len(manifest['documents']),
            'backup': str(backup_dir) if backup_dir else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Existing working project folder')
    parser.add_argument('--destination', default=PROJECT_ROOT, type=Path)
    parser.add_argument('--overwrite', action='store_true', help='Back up and replace existing same-collection state')
    args = parser.parse_args()
    try:
        result = restore(args.source, args.destination, overwrite=args.overwrite)
    except (OSError, ValueError, sqlite3.Error, TimeoutError) as exc:
        parser.exit(1, f'State restore stopped: {exc}\n')
    print('Local state restored. Source project and Pinecone vectors unchanged.')
    print(f"Index: {result['index']} | Namespace: {result['namespace']} | Documents: {result['documents']}")
    for name in result['copied']:
        print('Copied:', name)
    for name in result['missing']:
        print('Not present in source (not fabricated):', name)
    if result['backup']:
        print('Previous destination state backed up:', result['backup'])
    print('Add your private .env separately. It was NOT copied. Do not publish the restored state.')


if __name__ == '__main__':
    main()

"""Create a public-source ZIP, excluding credentials and private working data.

Only known project folders and safe file types are included. This also scans
included plain text for recognizable credential patterns. Inspect screenshots and
documents manually before publication; this is not a universal secret detector.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {'rag', 'scripts', 'tests', 'evaluation', 'docs', 'screenshots', 'sample_documents', '.streamlit'}
ROOT_FILES = {'app.py', 'README.md', 'START_HERE.md', '.env.example', '.gitignore',
              'check_gemini.py', 'check_pinecone.py', 'run_windows.bat', 'SHA256SUMS.txt'}
EXTENSIONS = {'.py', '.md', '.txt', '.toml', '.ps1', '.bat', '.csv', '.png', '.jpg', '.jpeg', '.svg', '.docx', '.pdf'}
SECRET_PATTERNS = [re.compile(p) for p in (
    r'AIza[0-9A-Za-z_-]{35}', r'pcsk_[0-9A-Za-z_-]{20,}',
    r'sk-lf-[0-9a-fA-F-]{30,}', r'ghp_[A-Za-z0-9]{30,}',
    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
)]


def is_public(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    parts = rel.parts
    if not path.is_file() or path.is_symlink():
        return False
    if any(p.startswith(('.venv', '.state_import_', 'migration_backup_')) or
           p in {'venv', 'env', '__pycache__', '.git', '.local_backups', '.pytest_cache', 'dist', 'models', '.cache'} for p in parts):
        return False
    name = path.name
    if name.startswith('.env') and name != '.env.example':
        return False
    if name == 'secrets.toml' or '.bak' in name or name.endswith('.backup'):
        return False
    if rel.as_posix() in {'runtime/README.txt', 'artifacts/README.md'}:
        return True
    if len(parts) == 1:
        return name in ROOT_FILES or (name.startswith('requirements') and name.endswith('.txt') and not name.startswith('requirements-installed'))
    if parts[0] not in FOLDERS or path.suffix.lower() not in EXTENSIONS:
        return False
    if parts[0] == 'evaluation' and any(name.startswith(p) for p in ('ragas_results', 'ragas_ready', 'groundlens_ragas_input')):
        return False
    if path.suffix.lower() == '.pdf':
        return parts[0] == 'sample_documents' and name.startswith('demo_')
    if path.suffix.lower() == '.docx':
        return rel.as_posix() == 'docs/report/C033_GroundLens_Final_Report.docx'
    return True


def public_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob('*') if is_public(p, root))


def build(root: Path, output: Path) -> tuple[int, str]:
    root, output = root.resolve(), output.resolve()
    paths = public_files(root)
    if not (root/'app.py').is_file() or not paths:
        raise ValueError('Project root is missing app.py or public files.')
    for path in paths:
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.pdf', '.docx'}:
            continue
        content = path.read_text(encoding='utf-8-sig')
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            raise ValueError(f'Possible credential detected in {path.relative_to(root)}. No ZIP created.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.zip.tmp')
    try:
        with ZipFile(temporary, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for path in paths:
                archive.write(path, 'GroundLens/'+path.relative_to(root).as_posix())
        with ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError('ZIP integrity check failed.')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return len(paths), digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dist/GroundLens_Repository.zip')
    args = parser.parse_args()
    count, digest = build(ROOT, args.output)
    print(f'Created {args.output.resolve()} ({count} files)')
    print('SHA256:', digest)
    print('Private .env, .venv*, runtime databases, fitted DSPy JSON and live answer exports excluded.')
    print('Review screenshot/document content before publishing. Extract and upload contents into the existing repository.')


if __name__ == '__main__':
    main()

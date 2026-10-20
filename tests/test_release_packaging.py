import json
from pathlib import Path
import sqlite3
from zipfile import ZipFile
import pytest
from scripts.restore_local_state import restore, _corpus_settings
from scripts.build_repo_zip import build, is_public
from evaluation.ragas_runner import cli_args


def make_source(tmp_path):
    src = tmp_path/'source'
    (src/'runtime').mkdir(parents=True)
    manifest = {'index':'demo-index','namespace':'demo-space','documents':{'doc':{'filename':'demo.pdf','chunks':1}}}
    (src/'runtime/pinecone_manifest.json').write_text(json.dumps(manifest))
    (src/'.env').write_text('PINECONE_INDEX=demo-index\nPINECONE_NAMESPACE=demo-space\nGEMINI_API_KEY=private-test-value\n')
    with sqlite3.connect(src/'runtime/groundlens_bm25.sqlite3') as db:
        db.execute('CREATE TABLE chunks (text TEXT)')
        db.execute('INSERT INTO chunks VALUES (?)',('Synthetic evidence',))
    (src/'artifacts').mkdir()
    (src/'artifacts/dspy_program.json').write_text('{"synthetic":"test-only"}')
    (src/'evaluation').mkdir()
    (src/'evaluation/ragas_results.csv').write_text('user_input,faithfulness\nSynthetic question,0.5\n')
    dest=tmp_path/'destination'
    dest.mkdir()
    (dest/'app.py').write_text('# test placeholder')
    return src, dest


def test_restore_preserves_source_and_does_not_copy_env(tmp_path):
    src, dest=make_source(tmp_path)
    before=(src/'runtime/pinecone_manifest.json').read_bytes()
    result=restore(src,dest)
    assert result['documents']==1
    assert not (dest/'.env').exists()
    assert (src/'runtime/pinecone_manifest.json').read_bytes()==before
    assert (dest/'artifacts/dspy_program.json').is_file()
    assert (dest/'evaluation/ragas_results.csv').is_file()
    with sqlite3.connect(dest/'runtime/groundlens_bm25.sqlite3') as db:
        assert db.execute('SELECT text FROM chunks').fetchone()[0]=='Synthetic evidence'


def test_restore_rejects_source_destination_same(tmp_path):
    src,_=make_source(tmp_path)
    with pytest.raises(ValueError, match='non-nested'):
        restore(src,src)


def test_restore_rejects_destination_namespace_mismatch(tmp_path):
    src,dest=make_source(tmp_path)
    (dest/'.env').write_text('PINECONE_NAMESPACE=different\n')
    with pytest.raises(ValueError,match='does not match'):
        restore(src,dest)
    assert not (dest/'runtime/pinecone_manifest.json').exists()


def test_restore_overwrite_requires_explicit_flag_and_backs_up(tmp_path):
    src,dest=make_source(tmp_path)
    restore(src,dest)
    with pytest.raises(FileExistsError):
        restore(src,dest)
    result=restore(src,dest,overwrite=True)
    assert (Path(result['backup'])/'runtime/pinecone_manifest.json').exists()


def test_restore_missing_artifact_does_not_fabricate_it(tmp_path):
    src,dest=make_source(tmp_path)
    (src/'artifacts/dspy_program.json').unlink()
    result=restore(src,dest)
    assert 'artifacts/dspy_program.json' in result['missing']
    assert not (dest/'artifacts/dspy_program.json').exists()


def test_restore_bad_json_fails_before_writing(tmp_path):
    src,dest=make_source(tmp_path)
    (src/'artifacts/dspy_program.json').write_text('not JSON')
    with pytest.raises(ValueError):
        restore(src,dest)
    assert not (dest/'runtime/pinecone_manifest.json').exists()


def test_corpus_settings_do_not_return_secrets(tmp_path):
    p=tmp_path/'.env'
    p.write_text('export PINECONE_INDEX="my-index" # comment\nPINECONE_NAMESPACE=my-ns # note\nGEMINI_API_KEY=secret\n')
    assert _corpus_settings(p)=={'PINECONE_INDEX':'my-index','PINECONE_NAMESPACE':'my-ns'}


@pytest.mark.parametrize('name',[
    '.env','.env.backup','.venv-ragas/Lib/library.py','runtime/pinecone_manifest.json',
    'runtime/groundlens_bm25.sqlite3','artifacts/dspy_program.json',
    'evaluation/ragas_results.csv','.streamlit/secrets.toml','migration_backup_old/.env',
    'evaluation/groundlens_ragas_input.csv',
])
def test_private_files_not_packaged(tmp_path,name):
    p=tmp_path/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('private')
    assert not is_public(p,tmp_path)


def test_public_archive_keeps_template_and_excludes_env(tmp_path):
    root=tmp_path/'project';root.mkdir()
    (root/'app.py').write_text('print("demo")')
    (root/'.env').write_text('SECRET=not-for-publication')
    (root/'.env.example').write_text('SECRET=YOUR_KEY')
    output=tmp_path/'public.zip'
    count,digest=build(root,output)
    with ZipFile(output) as z:
        assert 'GroundLens/.env.example' in z.namelist()
        assert 'GroundLens/.env' not in z.namelist()
        assert z.testzip() is None
    assert count==2 and len(digest)==64


def test_ragas_timeout_regression():
    path=Path(__file__).parents[1]/'evaluation/ragas_runner.py'
    assert 'timeout=240' in path.read_text()
    assert 'timeout=90' not in path.read_text()

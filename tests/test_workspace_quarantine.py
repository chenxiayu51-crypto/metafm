import importlib.util
import json
from pathlib import Path
import pytest
import os
import subprocess

spec=importlib.util.spec_from_file_location('quarantine',Path(__file__).parents[1]/'scripts/workspace_quarantine.py')


def _module():
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_quarantine_restores_exact_original_path_and_hash(tmp_path):
    module=_module();legacy=tmp_path/'legacy';source=legacy/'build/lib';source.mkdir(parents=True)
    (source/'example.py').write_text('print(42)',encoding='utf-8')
    target=tmp_path/'quarantine'
    mapping=module.quarantine_build(legacy,target)
    assert not (legacy/'build').exists()
    assert (target/'build/lib/example.py').read_text()=='print(42)'
    assert json.loads(mapping.read_text())['files'][0]['sha256']
    module.restore(mapping)
    assert (source/'example.py').read_text()=='print(42)'
    assert not (target/'build').exists()


def test_quarantine_refuses_data_or_checkpoints(tmp_path):
    module=_module();source=tmp_path/'legacy/build';source.mkdir(parents=True)
    (source/'unique.ckpt').write_bytes(b'unique')
    with pytest.raises(ValueError,match='protected'):
        module.quarantine_build(source.parent,tmp_path/'quarantine')
    assert (source/'unique.ckpt').exists()


def test_restore_never_overwrites_existing_source(tmp_path):
    module=_module();source=tmp_path/'legacy/build';source.mkdir(parents=True)
    (source/'x.py').write_text('original')
    mapping=module.quarantine_build(source.parent,tmp_path/'quarantine')
    source.mkdir();(source/'x.py').write_text('new')
    with pytest.raises(FileExistsError):module.restore(mapping)
    assert (source/'x.py').read_text()=='new'


@pytest.mark.skipif(os.name!='nt',reason='Windows junction regression')
def test_restore_refuses_redirected_original_parent(tmp_path):
    module=_module();legacy=tmp_path/'legacy';source=legacy/'build';source.mkdir(parents=True)
    (source/'x.py').write_text('original')
    mapping=module.quarantine_build(legacy,tmp_path/'quarantine')
    legacy.rename(tmp_path/'legacy_preserved');external=tmp_path/'external';external.mkdir()
    subprocess.run(['cmd','/c','mklink','/J',str(legacy),str(external)],check=True,capture_output=True)
    with pytest.raises(ValueError,match='redirect|reparse|link'):
        module.restore(mapping)
    assert not (external/'build').exists()


def test_normal_relative_path_is_not_a_redirect(tmp_path,monkeypatch):
    module=_module();legacy=tmp_path/'legacy';(legacy/'build').mkdir(parents=True)
    (legacy/'build/x.py').write_text('source');work=tmp_path/'work';work.mkdir()
    monkeypatch.chdir(work)
    mapping=module.quarantine_build('../legacy','quarantine')
    module.restore(mapping)
    assert (legacy/'build/x.py').read_text()=='source'

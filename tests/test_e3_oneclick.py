
import csv
import importlib.util
from pathlib import Path

import h5py
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "e3_oneclick.py"
spec = importlib.util.spec_from_file_location("e3v3", SCRIPT)
e3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e3)


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "canonical"
    manifest = e3.create_synthetic(root)
    corpus = e3.find_nature_corpus(root, None)
    return root, manifest, corpus


def test_five_corpora_with_train_val_intervals(fixture):
    root, manifest, corpus = fixture
    report = e3.audit(root, manifest, corpus, inspect_all=True)
    assert report["checked_h5"] == 12
    assert all(report["counts"][name]["train"] for name in e3.TASKS)
    assert all(report["counts"][name]["val"] for name in e3.TASKS)


def test_nature_unique_basenames(fixture):
    root, manifest, corpus = fixture
    with manifest.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    names = [Path(r["source_file"]).name for r in rows if r["dataset_id"] in e3.NATURE_DATASETS]
    assert len(names) == len(set(names))


def test_nature_corpus_has_six_half_open_intervals(fixture):
    root, manifest, corpus = fixture
    rows = e3.read_nature_corpus(corpus)
    assert len(rows) == 6
    assert {(r["start"],r["end"]) for r in rows} == {("0.0","4.0")}


def test_corpus_missing_fail_closed(fixture):
    root, manifest, corpus = fixture
    corpus.unlink()
    with pytest.raises(e3.E3Error, match="Nature corpus not found"):
        e3.find_nature_corpus(root, None)


def test_corpus_overlap_rejected(fixture):
    root, manifest, corpus = fixture
    with corpus.open(encoding="utf-8", newline="") as f:
        rdr=csv.DictReader(f)
        rows=list(rdr)
        cols=rdr.fieldnames
    r=dict(rows[0], start="1.0", end="2.0")
    rows.append(r)
    with corpus.open("w", encoding="utf-8", newline="") as f:
        wr=csv.DictWriter(f,fieldnames=cols)
        wr.writeheader();wr.writerows(rows)
    with pytest.raises(e3.E3Error,match="Overlapping Nature intervals"):
        e3.read_nature_corpus(corpus)


def test_missing_time_origin_rejected(fixture):
    root, manifest, corpus = fixture
    with manifest.open(encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    r=next(r for r in rows if r["dataset_id"] in e3.NATURE_DATASETS)
    with h5py.File(root/r["path"],"a") as h:
        del h["meta/source_json"]
    with pytest.raises(e3.E3Error,match="meta/source_json"):
        e3.audit(root, manifest, corpus, inspect_all=True)


def test_nature_official_unknown_uses_corpus(fixture):
    root, manifest, corpus = fixture
    with manifest.open(encoding="utf-8",newline="") as f:
        rd=csv.DictReader(f)
        rows=list(rd);columns=rd.fieldnames
    for r in rows:
        if r["dataset_id"] in e3.NATURE_DATASETS:
            r["official_split"]="unknown"
            r["split"]=""
            with h5py.File(root/r["path"],"a") as h:
                h.attrs["official_split"]="unknown"
    with manifest.open("w",encoding="utf-8",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=columns)
        wr.writeheader();wr.writerows(rows)
    report=e3.audit(root, manifest, corpus, inspect_all=True)
    assert all(report["counts"][ds]["train"] for ds in e3.NATURE_DATASETS)


def test_local_fixture_version_and_metadata(fixture):
    root, manifest, corpus = fixture
    with h5py.File(root/"gni_wrist/wrist_angle/gni_wrist_train_user/gni_wrist_train_session_right.h5","r") as h:
        assert h.attrs["schema_version"] == "1.0"
        assert h.attrs["source_file"]
        assert "source_time_origin_s" in h["meta/source_json"].asstr()[()]
    assert e3.create_synthetic(root) == manifest


def test_server_rejects_synthetic_root(fixture, tmp_path):
    root, manifest, corpus = fixture
    with pytest.raises(e3.E3Error,match="Synthetic"):
        e3.main(["server","--data-root",str(root),"--output",str(tmp_path)])


def test_without_corpus_audit_fails(fixture):
    root, manifest, corpus=fixture
    with pytest.raises(e3.E3Error,match="require a real corpus_path"):
        e3.audit(root,manifest,None)

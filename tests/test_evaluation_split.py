import pytest
from semg_fm.data.datamodule import CanonicalDataModule


def test_evaluation_val_does_not_relabel_or_require_train(canonical_meta_root):
    dm=CanonicalDataModule(manifest_path=canonical_meta_root/'manifest.csv',root=canonical_meta_root,
                           window_ms=200,stride_ms=100,batch_size=2,evaluation_split='val')
    dm.setup('test')
    assert len(dm.test_dataset)==0
    assert dm.train_dataset is None
    assert dm.test_dataset is not dm.val_dataset


def test_evaluation_cannot_select_train(canonical_meta_root):
    with pytest.raises(ValueError,match='evaluation_split'):
        CanonicalDataModule(manifest_path=canonical_meta_root/'manifest.csv',window_ms=200,
                            stride_ms=100,batch_size=2,evaluation_split='train')

import pytest


def test_verification_status_cannot_turn_missing_data_into_pass(tmp_path):
    from semg_fm.local_verification import VerificationRun
    run=VerificationRun(tmp_path)
    run.record('POSE_TRAINING','BLOCKED',dict(reason='Missing official Pose Train data'))
    with pytest.raises(ValueError): run.record('invalid','SKIPPED',{})
    run.write_summary()
    assert 'BLOCKED' in (tmp_path/'summary.md').read_text(encoding='utf-8')
    assert run.statuses['POSE_TRAINING']['status']=='BLOCKED'

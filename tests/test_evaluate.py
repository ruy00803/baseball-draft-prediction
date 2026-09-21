import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluate import SEED, NUMERIC, CATEGORIES, encode_targets, target_statistics, prepare, evaluate, validate


def sample():
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({name: rng.uniform(1, 100, 80) for name in NUMERIC})
    for name in CATEGORIES:
        frame[name] = rng.choice(['a', 'b', 'c'], len(frame))
    frame['Id'] = np.arange(len(frame))
    frame['Drafted'] = np.tile([0, 1], 40)
    frame.loc[0, 'Age'] = np.nan
    return frame


def test_validation_labels_and_unknown_categories_do_not_enter_training():
    frame = sample()
    train, valid = frame.iloc[:60].copy(), frame.iloc[60:].copy()
    valid['School'] = 'unseen_school'
    valid['Position_Type'] = 'unseen_type'
    valid['Age'] = np.nan
    first = prepare(train, train['Drafted'], valid, True)
    valid['Drafted'] = 1 - valid['Drafted']
    second = prepare(train, train['Drafted'], valid, True)
    for a, b in zip(first[:2], second[:2]):
        np.testing.assert_array_equal(a, b)
        assert np.isfinite(a).all()
    te = target_statistics(train, train['Drafted'], valid)
    np.testing.assert_allclose(te['School_TE'], train['Drafted'].mean())


def test_inner_oof_uses_only_inner_training_statistics():
    frame = sample()
    y = frame['Drafted']
    encoded, _ = encode_targets(frame, y, frame.iloc[:3])
    for fit, hold in StratifiedKFold(5, shuffle=True, random_state=SEED).split(frame, y):
        expected = target_statistics(frame.iloc[fit], y.iloc[fit], frame.iloc[hold])
        np.testing.assert_allclose(encoded.iloc[hold], expected)
    # 各カテゴリが1行のみなら、その行の正解を含む統計量は存在しない。
    frame['School'] = frame['Id'].astype(str)
    encoded, _ = encode_targets(frame, y, frame.iloc[:3])
    np.testing.assert_allclose(encoded['School_TE'], 0.5)


def test_full_evaluation_outputs(tmp_path):
    report = evaluate(sample(), tmp_path)
    assert len(report['results']) == 4
    assert report['results'][0]['fold_auc'] == [0.5] * 5
    baseline = report['results'][1]['mean_auc']
    for row in report['results']:
        assert len(row['fold_auc']) == 5
        assert row['delta_vs_lightgbm_baseline'] == pytest.approx(row['mean_auc'] - baseline)
    oof = pd.read_csv(tmp_path / 'oof_predictions.csv')
    assert len(oof) == 80 and not oof.isna().any().any()
    assert (tmp_path / 'feature_importance.png').stat().st_size > 0
    assert (tmp_path / 'metrics.json').exists()


def test_invalid_target_rejected():
    frame = sample()
    frame.loc[0, 'Drafted'] = 3
    with pytest.raises(ValueError, match='Drafted'):
        validate(frame)

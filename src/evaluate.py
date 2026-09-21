"""Compare fixed baselines and fold-local target encoding on identical folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

SEED = 42
CATEGORIES = ['School', 'Position', 'Position_Type', 'Player_Type']
NUMERIC = ['Year', 'Age', 'Height', 'Weight', 'Bench_Press_Reps', 'Vertical_Jump',
           'Broad_Jump', 'Sprint_40yd', 'Shuttle', 'Agility_3cone']


def validate(frame):
    required = ['Id', 'Drafted', *CATEGORIES, *NUMERIC]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if frame['Id'].isna().any() or frame['Id'].duplicated().any():
        raise ValueError('Id must be present and unique.')
    if frame['Drafted'].isna().any() or set(frame['Drafted'].unique()) != {0, 1}:
        raise ValueError('Drafted must contain both 0 and 1, with no missing values.')
    if frame['Drafted'].value_counts().min() < 10:
        raise ValueError('Each class needs at least 10 rows for nested 5-fold encoding.')
    for name in NUMERIC:
        if not pd.api.types.is_numeric_dtype(frame[name]):
            raise ValueError(f'{name} must be numeric.')
        if np.isinf(frame[name].dropna()).any():
            raise ValueError(f'{name} contains infinity.')
    if (frame['Height'].dropna() <= 0).any():
        raise ValueError('Height must be positive.')


def basic_features(frame, engineered):
    # 明示した列だけを使い、Id・Drafted・未知の列の混入を防ぐ。
    result = frame[NUMERIC + CATEGORIES].copy()
    if engineered:
        result['Age_missing'] = result['Age'].isna().astype(int)
        result['Power_Index'] = result['Bench_Press_Reps'] * result['Weight']
        result['Jump_Index'] = result['Vertical_Jump'] + result['Broad_Jump']
        result['Speed_Agility_Index'] = result['Sprint_40yd'] + result['Shuttle'] + result['Agility_3cone']
        # 単位が未確認のためBMIとは呼ばない。
        result['Weight_Height_Ratio'] = result['Weight'] / result['Height'] ** 2
    return result


def target_statistics(train, y, apply):
    """Use labels from train only. Unknown categories fall back to the prior."""
    output = pd.DataFrame(index=apply.index)
    prior = float(y.mean())
    for column in ['School', 'Position']:
        values = train[column].fillna('__MISSING__').astype(str)
        stats = pd.DataFrame({'category': values.to_numpy(), 'target': y.to_numpy()})
        grouped = stats.groupby('category')['target'].agg(['sum', 'count'])
        # 固定の擬似件数20で少数カテゴリを全体平均に寄せる。
        means = (grouped['sum'] + 20.0 * prior) / (grouped['count'] + 20.0)
        keys = apply[column].fillna('__MISSING__').astype(str)
        output[column + '_TE'] = keys.map(means).fillna(prior)
    return output


def encode_targets(train, y, valid):
    """Inner OOF for training rows; whole outer-training fold for validation."""
    train = train.reset_index(drop=True)
    y = y.reset_index(drop=True)
    encoded = pd.DataFrame(index=train.index, columns=['School_TE', 'Position_TE'], dtype=float)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fit_idx, hold_idx in inner.split(train, y):
        encoded.loc[hold_idx] = target_statistics(train.iloc[fit_idx], y.iloc[fit_idx], train.iloc[hold_idx])
    return encoded, target_statistics(train, y, valid)


def prepare(train, y, valid, engineered):
    a = basic_features(train.reset_index(drop=True), engineered)
    b = basic_features(valid.reset_index(drop=True), engineered)
    if engineered:
        train_te, valid_te = encode_targets(a, y, b)
        for name in train_te.columns:
            a[name], b[name] = train_te[name], valid_te[name]
        for frame in [a, b]:
            for source in ['Sprint_40yd', 'Bench_Press_Reps', 'Shuttle', 'Weight_Height_Ratio']:
                frame[source + '_Position_TE'] = frame[source] * frame['Position_TE']
        a, b = a.drop(columns=['School', 'Position']), b.drop(columns=['School', 'Position'])
    categorical = [column for column in CATEGORIES if column in a]
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    a[categorical] = encoder.fit_transform(a[categorical].fillna('__MISSING__').astype(str))
    b[categorical] = encoder.transform(b[categorical].fillna('__MISSING__').astype(str))
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    return imputer.fit_transform(a), imputer.transform(b), a.columns.tolist()


def evaluate(frame, output):
    validate(frame)
    output.mkdir(parents=True, exist_ok=True)
    y = frame['Drafted'].astype(int).reset_index(drop=True)
    frame = frame.reset_index(drop=True)
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(frame, y))
    rows, predictions, importance = [], pd.DataFrame({'Id': frame['Id'], 'Drafted': y}), []
    configs = [('constant_baseline', False), ('lightgbm_baseline', False),
               ('lightgbm_features', True), ('xgboost_features', True)]
    for name, engineered in configs:
        oof, scores = np.zeros(len(y)), []
        for fold, (fit, hold) in enumerate(splits, start=1):
            if name == 'constant_baseline':
                pred = np.full(len(hold), y.iloc[fit].mean())
            else:
                a, b, features = prepare(frame.iloc[fit], y.iloc[fit], frame.iloc[hold], engineered)
                params = dict(n_estimators=300, learning_rate=0.03, max_depth=4,
                              random_state=SEED, n_jobs=1)
                model = (XGBClassifier(**params, eval_metric='auc', tree_method='hist')
                         if name.startswith('xgboost') else
                         LGBMClassifier(**params, num_leaves=15, verbosity=-1, importance_type='gain'))
                model.fit(a, y.iloc[fit])
                pred = model.predict_proba(b)[:, 1]
                if name == 'lightgbm_features':
                    gain = model.feature_importances_.astype(float)
                    gain = gain / gain.sum() if gain.sum() else gain
                    importance.extend({'fold': fold, 'feature': f, 'gain_share': float(v)}
                                      for f, v in zip(features, gain))
            oof[hold] = pred
            scores.append(float(roc_auc_score(y.iloc[hold], pred)))
        predictions[name] = oof
        rows.append({'model': name, 'fold_auc': scores, 'mean_auc': float(np.mean(scores)),
                     'std_auc': float(np.std(scores)), 'pooled_oof_auc': float(roc_auc_score(y, oof))})
    baseline = rows[1]['mean_auc']
    for row in rows:
        row['delta_vs_lightgbm_baseline'] = row['mean_auc'] - baseline
    report = {'seed': SEED, 'folds': 5, 'rows': len(frame), 'positive_rate': float(y.mean()),
              'python': platform.python_version(),
              'versions': {lib: version(lib) for lib in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost']},
              'results': rows}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    predictions.to_csv(output / 'oof_predictions.csv', index=False)
    pd.DataFrame(importance).to_csv(output / 'feature_importance.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values = pd.DataFrame(importance).groupby('feature')['gain_share'].mean().sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(11, 8), layout='constrained')
    values.plot.barh(ax=ax, color='#237c89')
    ax.set(xlabel='Mean normalized gain across 5 folds', ylabel='', title='LightGBM feature importance')
    fig.savefig(output / 'feature_importance.png', dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=Path('data/train.csv'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    evaluate(pd.read_csv(args.train), args.output)

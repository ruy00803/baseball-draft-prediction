# 評価結果の出力先

`python src/evaluate.py --train data/train.csv --output results`で生成します。

- `metrics.json`：fold別・平均・標準偏差（ddof=0）・OOF AUC、ベースラインとの差、件数、指名率、環境。
- `oof_predictions.csv`：Id、正解、各モデルのOOF予測。選手単位の情報を含むため公開前に確認してください。
- `feature_importance.csv`：LightGBMのfold別の正規化gain。
- `feature_importance.png`：5-fold平均の正規化gain上位20特徴量。

元データが未配置のため、修正後の実測スコアはまだありません。生成ファイルはGitの追跡対象から外しています。

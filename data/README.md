# 入力データ

分析用に提供された架空の野球選手データを使用します。実在する選手や特定リーグの記録ではありません。元データは同梱していないため、再実行には課題で提供された同じCSVが必要です。

利用権限のある元の学習CSVを`data/train.csv`に配置してください。旧NotebookはColab上の`/content/train (1).csv`と`/content/test (1).csv`を読んでいました。新しいCV評価にtest.csvは不要です。

必要な列：

- `Id`：欠損・重複のないレコードID。モデルには入力しません。
- `Drafted`：指名有無の0/1。両クラスが必要です。
- `School`, `Position`, `Position_Type`, `Player_Type`：カテゴリ。
- `Year`, `Age`, `Height`, `Weight`, `Bench_Press_Reps`, `Vertical_Jump`, `Broad_Jump`, `Sprint_40yd`, `Shuttle`, `Agility_3cone`：数値。欠損を許容しますが、Heightの非欠損値は正の数とします。

余分な列は学習に使いません。内側・外側の層化5分割のため、各クラスに最低10件必要です。件数と指名率は評価時に`results/metrics.json`に出力します。

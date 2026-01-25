# CMI-Detect-Befavio-With-Sensor-Data
腕につけたウェアラブルデバイスのセンサーデータから、その人がどんな動きをしているかを分類するコンペティション
## ファイル構成
src/models.py　# モデルアーキテクチャ用のスクリプト
src/preprocess.py # 前処理用のスクリプト
src/train.py 訓練用のスクリプト
src/utils.py
src/config.py Hydraのデフォルト値はconfig,py記述するものとする
その他 src内に適宜追加してもよい

## 訓練アプローチ
* データセットは、8151個の時系列データで構成される。sequence_idがシーケンスを決める
* fold分けはgloupkfoldでsubject毎
* 特徴量エンジニアリングを行ってから、モデルに入れる。データリークには気をつけて。scaler等も保存する必要があるかも？？
* ハイパーパラメータはHydraを使って参照
* Pytorchを使ったニューラルネットワーク
* 評価は(二値分類 + 多クラス分類 ) / 2になりますが、訓練は多クラス分類だけにフォーカスすればいいです。
* Optimizerは、Adam, AdamW, RAdamScheduleFreeからhydraで選べるようにして、デフォルトはRAdamScheduleFree
    * RAdamScheduleFreeについては詳しくはこっちを調べて　https://github.com/facebookresearch/schedule_free
* Schedulerはwarmup + cosineアニーリングがデフォルトで
* timmのEMA V3を利用すること 0.99がデフォルトで

## モデルアーキテクチャ
* 特徴量毎にブランチ戦略をとる。特徴量のグループごとにConv1DBlockを作り、後に統合する
* 統合した特徴量をGRU / LSTM / Transformerに入れ、GolobalAveragePoolingを行う
* Poolingした特徴量をMLPに入れる

## 前処理
* 前処理クラスPreprocessorを作成する。
* 訓練データにfit_transformを行ったあと、検証データにtransformを行うことで前処理を行う
* 正規化はstanderd scaler

## 基本原則
* 実験管理はhydra, wandb, loggerを持ち入る
* wandbはエポックごとに、lr, loss, 他クラス分類スコア, 初期家事
* 出力ファイルはcfg.env.exp_output_dirへ
* 実行は docker-compose run --rm kaggle python experiments/expXXX/run.py
    * これがだめなら、uv run python experiments/expXXX/run.py
    * もしくはmake bash; python experiments/expXXX/run.py

## inputファイル
cfg.env.input_dir / 
train.csv
train_demographics.csv
test.csv
test_demographics.csv

## コンペの説明
https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data
かならず参照すること
## データセットの説明
https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/data
かならず参照すること
## 本コンペの評価指標
The evaluation metric for this contest is a version of macro F1 that equally weights two components:

Binary F1 on whether the gesture is one of the target or non-target types.
Macro F1 on gesture, where all non-target sequences are collapsed into a single non_target class
The final score is the average of the binary F1 and the macro F1 scores.

If your submission includes a gesture value not found in the train set your submission will trigger an error.
https://www.kaggle.com/code/metric/cmi-2025

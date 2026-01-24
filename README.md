# MLコンペ用実験テンプレート

## 特徴
- Docker によるポータブルなKaggleと同一の環境
- Hydra による実験管理
- 実験用スクリプトファイルを major バージョンごとにフォルダごとに管理 & 実験パラメータ設定を minor バージョンとしてファイルとして管理
   - 実験用スクリプトと実験パラメータ設定を同一フォルダで局所的に管理して把握しやすくする
- dataclass を用いた config 定義を用いることで、エディタの補完機能を利用できるように
## フォーク源から追加した機能
- 実験管理が容易に
  - ```experiments/{major_exp_name}```単位での実験管理
  - ```make bash exp={major_exp_name}``` 機能を追加し、```/kaggle/working```内で実験を完結
  - モデル、ソースコード、実験ログを一括でKaggle Datasetにして実験を管理
  - ```tools/upload_dataset.py```で一括データセット化
- 環境変数管理
  - ```.env.example```を追加
    - 環境変数にてAPI_KEY等を管理
    - それに伴い```compose.yaml```を修正

### 変更コードまとめ
* 実験スクリプト
    * ```experiments/exp000_sample```直下のディレクトリ構造
    * ```experiments/exp000_sample/run.py```
* 環境構築
  * ```Dockerfile```
  * ```Dockerfile.cpu```
  * ```compose.yaml```
  * ```compose.cpu.yaml```
  * ```.env.example```
* その他
  * ```tools/upload_dataset.py```

### Hydra による Config 管理
- Config は yamlとdictで定義するのではなく、dataclass を用いて定義することで、エディタの補完などの機能を使いつつタイポを防止できるようにする
- 各スクリプトに共通する環境依存となる設定は utils/env.py の EnvConfig で定義される
- 各スクリプトによって変わる設定は、実行スクリプトのあるフォルダ(`{major_exp_name}`)の中に `exp/{minor_exp_name}.yaml` として配置することで管理。
    - 実行時に `exp={minor_exp_name}` で上書きする
    - `{major_exp_name}` と `{minor_exp_name}` の組み合わせで実験が再現できるようにする

## Structure(罫線表記)
```text
.
├── experiments
│    └── exp000_sample
│        ├── exp
│        ├── output
│        ├── utils
│        ├── src
│        └── run.py
├── input
├── notebook
├── tools
├── output
├── Dockerfile
├── Dockerfile.cpu
├── LICENSE
├── Makefile
├── README.md
├── .env.example
├── compose.cpu.yaml
└── compose.yaml

```
## 環境変数の設定
```sh
cp .env.example .env
```
を行い、必要事項を記入
## Docker による環境構築
Dockerが利用できない方は、同md下部のuvによる環境設定を参照してください。
```sh
# imageのbuild
make build

# bash に入る場合
make bash
make bash exp={major_exp_number} # ex) exp=000

# jupyter lab を起動する場合
make jupyter

# CPUで起動する場合はCPU=1やCPU=True などをつける
```
### スクリプトの実行方法

```sh
make bash exp=000
python run.py
python run.py exp=001
```
もしくは
```sh
make bash
python experiments/exp000_sample/run.py exp=001
```

## Kaggle データセットの作成
```sh
# Kaggle API Keyが必要
# major_virsion_nameでそのまま提出
# -t: タイトル, -d: ディレクトリ
python tools/upload_dataset.py -t exp000 -d experiments/{major_virsion_name}
```

## (Dockerが使えない方向け)uvによる環境構築
### uvのインストール
詳しくは[こちら](https://docs.astral.sh/uv/getting-started/installation/v)を参照
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv version # インストールの確認
```
### .venv 仮想環境の作成
```sh
uv sync
```
適宜追加したいモジュールは、以下のコマンドで追加してください。その他利用方法は
```sh
uv add numpy
```
その他利用方法は[こちら](https://docs.astral.sh/uv/)を参照してください。

**注意事項**
* pythonバージョン, numpyバージョンはkaggle kernelに合わせ最新版ではない
  * ```python==3.11.13```
  * ```numpy==1.26.4```
* ```torch, cuda```は、各環境に合わせインストールしてください。
### スクリプトの実行方法
```sh
uv run experiments/exp000_sample/run.py exp=001
```
### jupyter notebook利用方法
uvによって作成された.venvを使って



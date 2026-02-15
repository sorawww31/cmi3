import json
import os
import shutil
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()


def copy_directory(source_dir: Path, dest_dir: Path):
    """
    source_dirの中身をすべてdest_dirにコピーする

    Args:
        source_dir: コピー元ディレクトリ
        dest_dir: コピー先ディレクトリ
    """
    # dest_dirが存在する場合は削除
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    # ディレクトリ全体をコピー
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=False)
    print(f"Copied {source_dir} to {dest_dir}")


def upload_single(dir: Path, title: str, user_name: str, new: bool):
    """単一のディレクトリをKaggleデータセットとしてアップロードする。"""
    if "_" in title:
        title = title.replace("_", "-")
    tmp_dir = Path("./tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    copy_directory(dir, tmp_dir)

    # dataset-metadata.jsonを作成
    dataset_metadata: dict[str, Any] = {}
    dataset_metadata["id"] = f"{user_name}/{title}"
    dataset_metadata["licenses"] = [{"name": "CC0-1.0"}]
    dataset_metadata["title"] = title

    with open(tmp_dir / "dataset-metadata.json", "w") as f:
        json.dump(dataset_metadata, f, indent=4)

    # api認証
    api = KaggleApi()
    api.authenticate()

    if new:
        api.dataset_create_new(
            folder=tmp_dir,
            dir_mode="tar",
            convert_to_csv=False,
            public=False,
        )
    else:
        api.dataset_create_version(
            folder=tmp_dir,
            version_notes="",
            dir_mode="tar",
            convert_to_csv=False,
        )

    # delete tmp dir
    shutil.rmtree(tmp_dir)
    print(f"Uploaded {dir} as '{title}'")


@click.command()
@click.option("--title", "-t", default="sorawww31-models")
@click.option("--dir", "-d", type=Path, default="./experiments")
@click.option("--user_name", "-u", default=os.getenv("KAGGLE_USERNAME"))
@click.option("--new", "-n", is_flag=True)
@click.option(
    "--exp",
    "-e",
    default=None,
    help="実験名 (例: exp007)。experiments/<exp>_* と output/<exp>_* を自動的にアップロードする。",
)
def main(
    title: str,
    dir: Path,
    user_name: str = "sorawww31",
    new: bool = False,
    exp: str | None = None,
):
    """dir以下のファイルをKaggleデータセットとしてアップロードする。

    --exp を指定すると、experiments/<exp>_* と output/<exp>_* に
    マッチするディレクトリを同じデータセットにまとめてアップロードする。
    データセット内では experiments/ と output/ のサブディレクトリに分かれる。

    Args:
        title (str): kaggleにアップロードするときのタイトル (--exp未指定時に使用)
        dir (Path): アップロードするファイルがあるディレクトリ (--exp未指定時に使用)
        user_name (str, optional): kaggleのユーザー名.
        new (bool, optional): 新規データセットとしてアップロードするかどうか.
        exp (str, optional): 実験名 (例: exp007).
    """
    if exp is not None:
        # experiments/<exp>_* と output/<exp>_* を探す
        base_dirs = [Path("./experiments"), Path("./output")]
        targets: list[Path] = []
        for base in base_dirs:
            targets.extend(sorted(base.glob(f"{exp}_*")))

        if not targets:
            print(
                f"Error: No directories matching '{exp}_*' found in experiments/ or output/"
            )
            return

        # データセットタイトルを最初に見つかったディレクトリ名から決定
        dataset_title = targets[0].name.replace("_", "-")

        # 全ターゲットを1つのtmpディレクトリにまとめる
        tmp_dir = Path("./tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Found {len(targets)} directories to merge into dataset '{dataset_title}':"
        )
        for target in targets:
            if not target.is_dir():
                print(f"  Skipping {target} (not a directory)")
                continue
            # experiments/exp007_scale -> tmp/experiments/
            dest = tmp_dir / target.parent.name
            shutil.copytree(target, dest)
            print(f"  - {target} -> {dest}")

        # dataset-metadata.jsonを作成
        if "_" in dataset_title:
            dataset_title = dataset_title.replace("_", "-")
        dataset_metadata: dict[str, Any] = {}
        dataset_metadata["id"] = f"{user_name}/{dataset_title}"
        dataset_metadata["licenses"] = [{"name": "CC0-1.0"}]
        dataset_metadata["title"] = dataset_title

        with open(tmp_dir / "dataset-metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=4)

        # api認証
        api = KaggleApi()
        api.authenticate()

        if new:
            api.dataset_create_new(
                folder=tmp_dir,
                dir_mode="tar",
                convert_to_csv=False,
                public=False,
            )
        else:
            api.dataset_create_version(
                folder=tmp_dir,
                version_notes="",
                dir_mode="tar",
                convert_to_csv=False,
            )

        shutil.rmtree(tmp_dir)
        print(f"Uploaded as '{dataset_title}'")
    else:
        # 従来の動作: --dir と --title を使う
        upload_single(dir, title, user_name, new)


if __name__ == "__main__":
    main()

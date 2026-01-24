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


@click.command()
@click.option("--title", "-t", default="sorawww31-models")
@click.option("--dir", "-d", type=Path, default="./experiments")
@click.option("--user_name", "-u", default=os.getenv("KAGGLE_USERNAME"))
@click.option("--new", "-n", is_flag=True)
def main(
    title: str,
    dir: Path,
    user_name: str = "sorawww31",
    new: bool = False,
):
    """extentionを指定して、dir以下のファイルをzipに圧縮し、kaggleにアップロードする。

    Args:
        title (str): kaggleにアップロードするときのタイトル
        dir (Path): アップロードするファイルがあるディレクトリ
        extentions (list[str], optional): アップロードするファイルの拡張子.
        user_name (str, optional): kaggleのユーザー名.
        new (bool, optional): 新規データセットとしてアップロードするかどうか.
    """
    tmp_dir = Path("./tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 拡張子が.pthのファイルをコピー
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


if __name__ == "__main__":
    main()

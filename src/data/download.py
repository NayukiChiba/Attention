"""
src/data/download.py

THUCNews数据集下载脚本

1. 从清华 NLP 官方源下载 THUCNews 数据集
2. 自动解压
"""

import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from config import paths

THUCNEWS_URL = [
    "http://thuctc.thunlp.org/THUCNews.zip",
    "https://github.com/skdjfla/THUCNews/archive/refs/heads/master.zip",
]

RAW_ZIP_PATH = paths.RAW_DATASETS_DIR / "THUCNews.zip"
EXTRACTED_DIR = paths.RAW_DATASETS_DIR / "THUCNews"


def download_file(url: str, save_path: Path) -> None:
    """
    下载文件并显示进度条
    Args:
        url(str): 文件下载链接
        save_path(Path): 下载后文件保存路径
    """
    print(f"正在下载数据集: {url}")

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()  # 检查请求是否成功

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 限制下载速度为1MB/s
    # write + binary模式打开文件
    with save_path.open("wb") as file:
        # unit="B"表示以字节为单位显示,unit_scale=True表示自动调整单位(KB, MB, GB等)
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="下载进度") as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                file.write(chunk)
                pbar.update(len(chunk))

    print(f"数据集下载完成: {save_path}")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """
    解压 zip 文件
    Args:
        zip_path(Path): zip 文件路径
        extract_dir(Path): 解压后文件保存目录
    """
    print(f"正在解压数据集: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"数据集解压完成: {extract_dir}")

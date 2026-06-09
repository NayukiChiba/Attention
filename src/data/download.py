"""
src/data/download.py

THUCNews数据集下载脚本

1. 从清华 NLP 官方源下载 THUCNews 数据集
2. 自动解压
"""

import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from config import paths

THUCNEWS_URL = [
    "http://thuctc.thunlp.org/THUCNews.zip",
    "https://thunlp.oss-cn-qingdao.aliyuncs.com/THUCNews.zip",
]


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
        # info_list()方法返回zip文件中所有成员的信息列表
        # namelist()方法返回zip文件中所有成员的名称列表
        members = zip_ref.infolist()
        # file_size属性表示成员的原始大小,通过sum()函数计算所有成员的总大小
        total_size = sum(member.file_size for member in members)
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="解压进度") as pbar:
            for member in members:
                # 修复中文乱码: cp437 -> utf-8
                try:
                    # ZIP中文件名被cp437错误解析，还原为UTF-8字节再正确解码
                    member.filename = member.filename.encode("cp437").decode("utf-8")
                except (UnicodeDecodeError, UnicodeEncodeError):
                    # 转换失败则保留原名
                    pass

                zip_ref.extract(member, extract_dir)
                pbar.update(member.file_size)
    print(f"数据集解压完成: {extract_dir}")


def download_thucnews() -> Path:
    """
    下载并解压 THUCNews 数据集
    """
    # 如果下载的数据集zip文件存在, 直接开始解压就可以
    if paths.DATASET_ZIP_PATH.exists():
        print(f"数据集zip文件已存在: {paths.DATASET_ZIP_PATH}")
    else:
        # 否则从网络下载
        for url in THUCNEWS_URL:
            try:
                download_file(url, paths.DATASET_ZIP_PATH)
                break  # 下载成功后跳出循环
            except Exception as e:
                print(f"下载失败: {url}, {e}")
        else:
            raise RuntimeError(
                "THUCNews数据集下载失败, 请手动下载:"
                "http://thuctc.thunlp.org/THUCNews.zip"
                f"到 {paths.DATASET_ZIP_PATH}中"
            )

    # 解压数据集到 raw 目录
    if paths.THUCNEWS_RAW_DIR.exists():
        # 清空已存在的解压目录
        shutil.rmtree(paths.THUCNEWS_RAW_DIR)

    # 解压到 raw 目录的父目录，让 THUCNews/ 自动创建
    extract_zip(paths.DATASET_ZIP_PATH, paths.RAW_DATASETS_DIR)

    return paths.THUCNEWS_RAW_DIR


if __name__ == "__main__":
    download_thucnews()

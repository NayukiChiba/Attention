"""
预处理, 数据整理
读取 THUCNews 数据集的原始文本文件, 将其转换为统一格式的文本文件, 以供后续处理使用

"""

import random
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from config import DataConfig, paths


def read_news_file(news_dir: Path) -> List[Tuple[str, str]]:
    """
    读取新闻文本文件, 返回标签和文本内容的列表
    Args:
        news_dir(Path): 新闻文本文件所在目录
    Returns:
        List[Tuple[str, str]]: (类别, 文本内容) 列表
    """
    data = []
    # 获取所有的新闻类别目录
    # d: Path对象
    categories = [dir for dir in news_dir.iterdir() if dir.is_dir()]
    print(f"正在读取数据集: {news_dir}, 共 {len(categories)} 个类别")

    for category in tqdm(categories, desc="读取新闻"):
        category_name = category.name  # Path对象的name属性返回最后一级目录名,即类别名
        # 获取该类别目录下的所有文本文件
        # glob("*.txt")方法返回该目录下所有以.txt结尾的文件路径生成器, list()函数将其转换为列表
        txt_files = list(category.glob("*.txt"))

        for txt_file in txt_files:
            # 读取文本文件内容
            try:
                with txt_file.open("r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:  # 只添加非空文本
                        data.append((category_name, content))
            except Exception as e:
                print(f"读取文件 {txt_file} 失败: {e}")
    print(f"数据集读取完成: {len(data)} 条数据")
    return data


def clean_text(text: str) -> str:
    """
    清洗文本内容, 去除多余的空白字符
    Args:
        text(str): 原始文本内容
    Returns:
        str: 清洗后的文本内容
    Example:
    >>> clean_text("  这是   一段   测试文本。  ")
    '这是 一段 测试文本。'
    """
    # 将连续的空白字符替换为一个空格, strip()方法去除首尾空白
    return " ".join(text.split())


def filter_data(
    data: List[Tuple[str, str]], data_config: DataConfig
) -> List[Tuple[str, str]]:
    """
    过滤数据, 去除文本内容过短的数据
    Args:
        data(List[Tuple[str, str]]): (类别, 文本内容) 列表
        data_config(DataConfig): 配置对象，包含过滤参数
    Returns:
        List[Tuple[str, str]]: 过滤后的数据列表
    """
    before_count = len(data)

    filtered_data = []

    for label, text in data:
        text_length = len(text)  # 计算文本长度

        # 长度过滤
        if text_length < data_config.min_text_length:
            continue  # 跳过过短的文本
        if text_length > data_config.max_text_length:
            text = text[: data_config.max_text_length]  # 截断过长的文本

        # 去除空文本
        if not text.strip():
            continue  # 跳过空文本

        filtered_data.append((label, text))

    after_count = len(filtered_data)
    print(f"数据过滤完成: {before_count} -> {after_count} 条数据")
    return filtered_data


def split_dataset(
    data: List[Tuple[str, str]], data_config: DataConfig
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    划分数据集为训练集、验证集和测试集
    Args:
        data(List[Tuple[str, str]]): (类别, 文本内容) 列表
        data_config(DataConfig): 配置对象，包含划分参数
    Returns:
        Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]: 训练集、验证集和测试集列表
    """
    # 打乱数据顺序以保证划分的随机性
    from collections import defaultdict

    # 按照类别分组数据
    label_groups = defaultdict(list)
    for label, text in data:
        label_groups[label].append((label, text))

    train_data, val_data, test_data = [], [], []

    # 对每个类别的数据进行划分
    random.seed(data_config.seed)  # 设置随机种子以保证划分可复现

    for label, items in label_groups.items():
        random.shuffle(items)  # 打乱该类别的数据顺序

        total = len(items)
        train_end = int(total * data_config.train_ratio)
        val_end = train_end + int(total * data_config.val_ratio)

        train_data.extend(items[:train_end])
        val_data.extend(items[train_end:val_end])
        test_data.extend(items[val_end:])

    # 再次打乱训练集、验证集和测试集的顺序
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    print(
        f"数据集划分完成: 训练集 {len(train_data)} 条, 验证集 {len(val_data)} 条, 测试集 {len(test_data)} 条"
    )
    return train_data, val_data, test_data


def save_dataset(
    train_data: List[Tuple[str, str]],
    val_data: List[Tuple[str, str]],
    test_data: List[Tuple[str, str]],
) -> None:
    """
    保存数据集到文件

    格式: 每行 "label\ttext"
    """
    paths.INTERIM_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    def write_file(path: Path, data: List[Tuple[str, str]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for label, text in data:
                # Tab 分隔，方便后续按需拆分
                f.write(f"{label}\t{text}\n")

    write_file(paths.INTERIM_TRAIN_DATASET_PATH, train_data)
    write_file(paths.INTERIM_VAL_DATASET_PATH, val_data)
    write_file(paths.INTERIM_TEST_DATASET_PATH, test_data)

    print("\n已保存:")
    print(f"  {paths.INTERIM_TRAIN_DATASET_PATH}")
    print(f"  {paths.INTERIM_VAL_DATASET_PATH}")
    print(f"  {paths.INTERIM_TEST_DATASET_PATH}")

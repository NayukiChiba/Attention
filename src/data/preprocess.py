"""
预处理, 数据整理
读取 THUCNews 数据集的原始文本文件, 将其转换为统一格式的文本文件, 以供后续处理使用

"""

from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm


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

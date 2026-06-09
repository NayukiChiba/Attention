"""
src/data/tokenizer.py
字符级分词器实现
"""

from typing import Dict, List

from config import DataConfig


class CharTokenizer:
    """
    字符级分词器

    特殊token
    - <PAD>: 填充token, 用于补齐序列到固定长度
    - <UNK>: 未知字符token, 用于表示词表中未出现的字符
    - <BOS>: 序列起始token, 用于标记文本序列的开始
    - <EOS>: 序列结束token, 用于标记文本序列的结束
    """

    def __init__(self, data_config):
        self.data_config = data_config or DataConfig()

        # 特殊token列表
        self.pad_token = self.data_config.pad_token
        self.unk_token = self.data_config.unk_token
        self.bos_token = self.data_config.bos_token
        self.eos_token = self.data_config.eos_token
        self.special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]

        # 初始化词表和反向词表
        self.char2id: Dict[str, int] = {}
        self.id2char: Dict[int, str] = {}
        # 从特殊token开始构建词表, id 固定为 0, 1, 2, 3
        for idx, token in enumerate(
            [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        ):
            self.char2id[token] = idx
            self.id2char[idx] = token

    def build_vocab(self, texts: List[str]) -> None:
        """
        从文本列表中构建词表
        Args:
            texts (List[str]): 训练文本列表

        """
        char_set = set()
        for text in texts:
            char_set.update(text)

        # 移除特殊token已经占用的id
        char_set -= set(self.char2id.keys())

        # 按照 Unicode 编码顺序排序字符, 确保每次构建词表的顺序一致
        sorted_chars = sorted(char_set)

        # 分配 id 从特殊token之后开始
        start_idx = len(self.char2id)
        for idx, char in enumerate(sorted_chars, start=start_idx):
            self.char2id[char] = idx
            self.id2char[idx] = char

        print(f"词表构建完成: {len(self.char2id)} 个字符 (包含特殊token)")
        print(f"特殊 token 有: {len(self.special_tokens)}个")
        print(f"普通字符有: {len(sorted_chars)}个")

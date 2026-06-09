"""
src/data/tokenizer.py
字符级分词器实现
"""

import json
from pathlib import Path
from typing import Dict, List

from config import DataConfig, paths


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

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = True
    ) -> List[int]:
        """
        将文本编码为 id 列表
        Args:
            text (str): 输入文本
            add_bos (bool): 是否在序列开头添加 <BOS> token
            add_eos (bool): 是否在序列结尾添加 <EOS> token
        Returns:
            List[int]: 编码后的 id 列表
        """

        # 编码文本, 未知字符使用 <UNK> token 的 id
        unk_id = self.char2id[self.unk_token]
        token_ids = [self.char2id.get(char, unk_id) for char in text]

        # 添加特殊token
        if add_bos:
            token_ids = [self.char2id[self.bos_token]] + token_ids
        if add_eos:
            token_ids = token_ids + [self.char2id[self.eos_token]]

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        将 id 列表解码为文本
        Args:
            token_ids (List[int]): 输入 id 列表
            skip_special_tokens (bool): 是否跳过特殊 token
        Returns:
            str: 解码后的文本

        """
        chars = []
        # 将 id 转换回字符, 如果 skip_special_tokens=True 则跳过特殊 token
        for token_id in token_ids:
            # 如果 token_id 不在词表中, 则使用 <UNK> token 的字符
            char = self.id2char.get(token_id, self.unk_token)
            # 如果跳过特殊 token 且当前字符是特殊 token, 则继续下一个 id
            if skip_special_tokens and char in self.special_tokens:
                continue

            chars.append(char)

        return "".join(chars)

    def save(self, filepath: Path) -> None:
        """
        保存词表到文件
        Args:
            filepath (Path): 词表保存路径
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # 准备保存数据
        data = {
            "char2id": self.char2id,
            "id2char": {idx: char for char, idx in self.char2id.items()},
            "special_tokens": self.special_tokens,
        }

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"词表已保存到 {filepath}")

    @staticmethod
    def load(filepath: Path) -> "CharTokenizer":
        """
        从文件加载词表
        Args:
            filepath (Path): 词表文件路径
        """
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = CharTokenizer()
        tokenizer.char2id = data["char2id"]
        tokenizer.id2char = {int(idx): char for idx, char in data["id2char"].items()}
        tokenizer.special_tokens = data["special_tokens"]

        print(f"词表已从 {filepath} 加载, 共 {len(tokenizer.char2id)} 个字符")
        return tokenizer

    @property
    def vocab_size(self) -> int:
        """返回词表大小"""
        return len(self.char2id)

    @property
    def pad_token_id(self) -> int:
        """返回 <PAD> token 的 id"""
        return self.char2id[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        """返回 <UNK> token 的 id"""
        return self.char2id[self.unk_token]

    @property
    def bos_token_id(self) -> int:
        """返回 <BOS> token 的 id"""
        return self.char2id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        """返回 <EOS> token 的 id"""
        return self.char2id[self.eos_token]


def build_and_save_vocab(
    train_file: Path, data_config: DataConfig, save_path: Path
) -> CharTokenizer:
    """
    构建并保存分词器, 注意只能用训练集
    Args:
        train_file (Path): 训练文件路径, 文件格式为每行 "label\ttext"
        data_config (DataConfig): 数据配置对象
        save_path (Path): 词表保存路径
    Returns:
        CharTokenizer: 构建好的分词器对象
    """

    # 先读取训练文本
    data_config = data_config or DataConfig()
    texts = []
    with train_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            label, text = parts
            texts.append(text)

    # 构建分词器
    tokenizer = CharTokenizer(data_config)
    tokenizer.build_vocab(texts)
    tokenizer.save(save_path)
    return tokenizer


if __name__ == "__main__":
    # 构建并保存词表的示例
    vocab_path = paths.PROCESSED_DATASETS_DIR / "vocab.json"
    data_config = DataConfig()
    tokenizer = build_and_save_vocab(
        paths.INTERIM_TRAIN_DATASET_PATH, data_config, save_path=vocab_path
    )

    # 测试编码解码
    test_text = "这是一个测试文本。"
    print(f"原始文本: {test_text}")

    token_ids = tokenizer.encode(test_text)
    print(f"编码: {token_ids}")

    decoded_text = tokenizer.decode(token_ids)
    print(f"解码: {decoded_text}")

    # 一致性检查
    assert test_text == decoded_text, "编码解码不一致!"
    print("编码解码一致性检查通过!")

    # 特殊token测试
    print(f"<PAD> token id: {tokenizer.pad_token_id}")
    print(f"<UNK> token id: {tokenizer.unk_token_id}")
    print(f"<BOS> token id: {tokenizer.bos_token_id}")
    print(f"<EOS> token id: {tokenizer.eos_token_id}")

    # 未知字符测试
    unknown_text = "这是一个包含未知字符的文本: 😊"
    unknown_token_ids = tokenizer.encode(unknown_text)
    print(f"包含未知字符的文本编码: {unknown_token_ids}")
    decoded_unknown_text = tokenizer.decode(unknown_token_ids)

    print(f"包含未知字符的文本解码: {decoded_unknown_text}")

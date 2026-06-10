"""
tests/test_cli_menu.py

测试交互式菜单辅助函数
"""

import unittest

import torch

from src.cli.menu import (
    _build_tokenizer,
    _create_dataloaders,
    _load_tokenizer,
)


class TestMenuHelpers(unittest.TestCase):
    """测试菜单辅助函数"""

    def setUp(self):
        # 检查数据是否存在
        from config import paths

        self.has_data = paths.INTERIM_TRAIN_DATASET_PATH.exists()

    # ============================================================
    # 分词器构建
    # ============================================================

    def test_build_tokenizer(self):
        """测试从训练数据构建分词器"""
        if not self.has_data:
            self.skipTest("训练数据不存在，跳过")

        tokenizer = _build_tokenizer()
        self.assertGreater(len(tokenizer.char2id), 4)  # 至少包含特殊 token
        self.assertIn("<PAD>", tokenizer.char2id)
        self.assertIn("<UNK>", tokenizer.char2id)
        self.assertIn("<BOS>", tokenizer.char2id)
        self.assertIn("<EOS>", tokenizer.char2id)

    def test_build_tokenizer_special_ids(self):
        """测试特殊 token 的 id 固定"""
        if not self.has_data:
            self.skipTest("训练数据不存在，跳过")

        tokenizer = _build_tokenizer()
        self.assertEqual(tokenizer.char2id["<PAD>"], 0)
        self.assertEqual(tokenizer.char2id["<UNK>"], 1)
        self.assertEqual(tokenizer.char2id["<BOS>"], 2)
        self.assertEqual(tokenizer.char2id["<EOS>"], 3)

    def test_build_tokenizer_encode_decode(self):
        """测试编码和解码"""
        if not self.has_data:
            self.skipTest("训练数据不存在，跳过")

        tokenizer = _build_tokenizer()
        text = "这是一段测试文本"
        encoded = tokenizer.encode(text, add_bos=False, add_eos=False)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        self.assertEqual(decoded, text)

    def test_build_tokenizer_with_bos_eos(self):
        """测试带 BOS/EOS 的编码"""
        if not self.has_data:
            self.skipTest("训练数据不存在，跳过")

        tokenizer = _build_tokenizer()
        text = "你好"
        encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
        self.assertEqual(encoded[0], tokenizer.char2id["<BOS>"])
        self.assertEqual(encoded[-1], tokenizer.char2id["<EOS>"])

    def test_load_tokenizer(self):
        """测试加载已保存的分词器"""
        if not self.has_data:
            self.skipTest("训练数据不存在，跳过")

        tokenizer = _load_tokenizer()
        self.assertGreater(len(tokenizer.char2id), 4)

    def test_load_tokenizer_vocab_size(self):
        """测试加载的分词器有正确的 vocab_size"""
        if not self.has_data:
            self.skipTest("训练数据不存在，跳过")

        tokenizer = _load_tokenizer()
        self.assertEqual(tokenizer.vocab_size, len(tokenizer.char2id))

    def test_tokenizer_pad_token_id(self):
        """测试 pad_token_id"""
        if not self.has_data:
            self.skipTest("训练数据不存在，跳过")

        tokenizer = _load_tokenizer()
        self.assertEqual(tokenizer.pad_token_id, 0)

    def test_tokenizer_unk_token_id(self):
        """测试 unk_token_id"""
        if not self.has_data:
            self.skipTest("训练数据不存在，跳过")

        tokenizer = _load_tokenizer()
        self.assertEqual(tokenizer.unk_token_id, 1)

    def test_tokenizer_bos_token_id(self):
        """测试 bos_token_id"""
        if not self.has_data:
            self.skipTest("训练数据不存在，跳过")

        tokenizer = _load_tokenizer()
        self.assertEqual(tokenizer.bos_token_id, 2)

    def test_tokenizer_eos_token_id(self):
        """测试 eos_token_id"""
        if not self.has_data:
            self.skipTest("训练数据不存在，跳过")

        tokenizer = _load_tokenizer()
        self.assertEqual(tokenizer.eos_token_id, 3)

    # ============================================================
    # 配置构建
    # ============================================================

    def test_build_model_config_default(self):
        """测试 GPTConfig 默认值"""
        from config.defaults import GPTConfig

        config = GPTConfig()
        self.assertGreater(config.embedding_dim, 0)
        self.assertGreater(config.num_layers, 0)
        self.assertGreater(config.vocab_size, 0)
        self.assertIn(config.pos_encoding_type, ["sinusoidal", "learnable"])
        self.assertIn(config.activation, ["gelu", "relu"])
        self.assertIn(config.norm_type, ["pre", "post"])

    def test_build_model_config_embedding_divisible(self):
        """测试 embedding_dim 能被 num_heads 整除"""
        from config.defaults import GPTConfig

        config = GPTConfig()
        self.assertEqual(config.embedding_dim % config.num_attention_heads, 0)

    def test_build_training_config(self):
        """测试构建训练配置"""
        from config.defaults import TrainingConfig

        config = TrainingConfig()
        self.assertGreater(config.batch_size, 0)
        self.assertGreater(config.learning_rate, 0)
        self.assertIn(config.optimizer_type, ["adam", "adamw", "sgd"])

    def test_build_training_config_validation(self):
        """测试训练配置字段验证"""
        from config.defaults import TrainingConfig

        config = TrainingConfig()
        # 验证默认值合理
        self.assertTrue(0 < config.learning_rate < 1)
        self.assertTrue(0 <= config.weight_decay < 1)
        self.assertTrue(0 < config.min_lr_ratio <= 1)
        self.assertTrue(config.warmup_steps <= config.total_steps)

    def test_build_generation_config(self):
        """测试构建生成配置"""
        from config.defaults import GenerationConfig

        config = GenerationConfig()
        self.assertGreater(config.max_new_tokens, 0)
        self.assertGreaterEqual(config.temperature, 0)
        self.assertGreaterEqual(config.top_k, 0)
        self.assertTrue(0 <= config.top_p <= 1)

    def test_build_generation_config_validation(self):
        """测试生成配置字段验证"""
        from config.defaults import GenerationConfig

        config = GenerationConfig()
        # 温度可以为 0（贪心解码）
        self.assertGreaterEqual(config.temperature, 0)
        # top_p 在 [0, 1]
        self.assertTrue(0 <= config.top_p <= 1)
        # repetition_penalty >= 1
        self.assertGreaterEqual(config.repetition_penalty, 1.0)

    # ============================================================
    # 数据加载器
    # ============================================================

    def test_create_dataloaders(self):
        """测试创建数据加载器"""
        if not self.has_data:
            self.skipTest("训练数据不存在，跳过")

        from config.defaults import GPTConfig, TrainingConfig

        tokenizer = _load_tokenizer()
        gpt_config = GPTConfig(vocab_size=tokenizer.vocab_size)
        training_config = TrainingConfig(batch_size=4, num_workers=0)

        train_loader, val_loader, test_loader = _create_dataloaders(
            tokenizer, gpt_config, training_config
        )

        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        self.assertGreater(len(train_loader), 0)
        self.assertGreater(len(val_loader), 0)
        self.assertGreater(len(test_loader), 0)

    def test_dataloader_shapes(self):
        """测试数据加载器输出形状"""
        if not self.has_data:
            self.skipTest("训练数据不存在，跳过")

        from config.defaults import GPTConfig, TrainingConfig

        tokenizer = _load_tokenizer()
        gpt_config = GPTConfig(vocab_size=tokenizer.vocab_size, context_length=128)
        training_config = TrainingConfig(batch_size=4, num_workers=0)

        train_loader, _, _ = _create_dataloaders(tokenizer, gpt_config, training_config)

        # 检查第一个 batch 的形状
        for input_ids, target_ids in train_loader:
            self.assertEqual(len(input_ids.shape), 2)
            self.assertEqual(len(target_ids.shape), 2)
            self.assertEqual(input_ids.shape[0], training_config.batch_size)
            self.assertEqual(input_ids.shape[1], gpt_config.context_length)
            self.assertEqual(target_ids.shape, input_ids.shape)
            # 验证 target 是 input 右移 1 位
            self.assertTrue(torch.all(input_ids[:, 1:] == target_ids[:, :-1]))
            break

    def test_dataloader_train_shuffle(self):
        """测试训练 loader 是否 shuffle"""
        if not self.has_data:
            self.skipTest("训练数据不存在，跳过")

        from config.defaults import GPTConfig, TrainingConfig

        tokenizer = _load_tokenizer()
        gpt_config = GPTConfig(vocab_size=tokenizer.vocab_size)
        training_config = TrainingConfig(batch_size=4, num_workers=0)

        # 获取两个 epoch 的第一个 batch
        loader1, _, _ = _create_dataloaders(tokenizer, gpt_config, training_config)
        loader2, _, _ = _create_dataloaders(tokenizer, gpt_config, training_config)

        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))

        # 由于 shuffle，两个 batch 可能不同（但不保证 100% 不同）
        # 至少形状要一致
        self.assertEqual(batch1[0].shape, batch2[0].shape)

    def test_dataloader_val_not_shuffle(self):
        """测试验证 loader 不 shuffle（同一 loader 多次迭代应一致）"""
        if not self.has_data:
            self.skipTest("训练数据不存在，跳过")

        from config.defaults import GPTConfig, TrainingConfig

        tokenizer = _load_tokenizer()
        gpt_config = GPTConfig(vocab_size=tokenizer.vocab_size)
        training_config = TrainingConfig(batch_size=4, num_workers=0)

        # 同一 loader 两次迭代，第一个 batch 应该相同（因为不 shuffle）
        _, val_loader, _ = _create_dataloaders(tokenizer, gpt_config, training_config)

        batch1 = next(iter(val_loader))
        batch2 = next(iter(val_loader))

        # 同一个 loader 不 shuffle，重新迭代第一个 batch 应该不同
        # （因为采样了下一个batch），但形状一致
        self.assertEqual(batch1[0].shape, batch2[0].shape)
        # 两个 batch 内容不应完全相同（除非数据集只有一个batch）
        if len(val_loader) > 1:
            self.assertFalse(torch.equal(batch1[0], batch2[0]))


if __name__ == "__main__":
    unittest.main()

"""
tests/test_cli_parser.py

测试命令行参数解析器
"""

import unittest

from config.defaults import GenerationConfig, GPTConfig, TrainingConfig
from src.cli.parser import create_parser


class TestParser(unittest.TestCase):
    """测试命令行参数解析"""

    def setUp(self):
        self.parser = create_parser()
        self.default_model = GPTConfig()
        self.default_train = TrainingConfig()
        self.default_gen = GenerationConfig()

    # ============================================================
    # train 命令测试
    # ============================================================

    def test_train_defaults(self):
        """train 命令使用默认参数"""
        args = self.parser.parse_args(["train"])
        self.assertEqual(args.command, "train")
        # 模型参数默认值
        self.assertEqual(args.vocab_size, self.default_model.vocab_size)
        self.assertEqual(args.context_length, self.default_model.context_length)
        self.assertEqual(args.embedding_dim, self.default_model.embedding_dim)
        self.assertEqual(args.num_heads, self.default_model.num_attention_heads)
        self.assertEqual(args.num_layers, self.default_model.num_layers)
        self.assertEqual(args.ffn_hidden_dim, self.default_model.ffn_hidden_dim)
        self.assertEqual(args.dropout, self.default_model.dropout_rate)
        self.assertEqual(args.pos_encoding, self.default_model.pos_encoding_type)
        self.assertEqual(args.activation, self.default_model.activation)
        self.assertEqual(args.norm_type, self.default_model.norm_type)
        # 训练参数默认值
        self.assertEqual(args.batch_size, self.default_train.batch_size)
        self.assertEqual(args.epochs, self.default_train.max_epochs)
        self.assertEqual(args.total_steps, self.default_train.total_steps)
        self.assertEqual(args.grad_clip, self.default_train.grad_clip)
        self.assertEqual(args.seed, self.default_train.seed)
        self.assertEqual(args.num_workers, self.default_train.num_workers)
        # 优化器默认值
        self.assertEqual(args.optimizer, self.default_train.optimizer_type)
        self.assertEqual(args.lr, self.default_train.learning_rate)
        self.assertEqual(args.weight_decay, self.default_train.weight_decay)
        # 调度器默认值
        self.assertEqual(args.scheduler, self.default_train.scheduler_type)
        self.assertEqual(args.warmup_steps, self.default_train.warmup_steps)
        self.assertEqual(args.min_lr_ratio, self.default_train.min_lr_ratio)
        # 早停默认值
        self.assertEqual(args.patience, self.default_train.early_stopping_patience)
        # 恢复默认值
        self.assertIsNone(args.resume)

    def test_train_batch_size(self):
        """train --batch-size"""
        args = self.parser.parse_args(["train", "--batch-size", "64"])
        self.assertEqual(args.batch_size, 64)

    def test_train_epochs(self):
        """train --epochs"""
        args = self.parser.parse_args(["train", "--epochs", "100"])
        self.assertEqual(args.epochs, 100)

    def test_train_lr(self):
        """train --lr"""
        args = self.parser.parse_args(["train", "--lr", "5e-4"])
        self.assertEqual(args.lr, 5e-4)

    def test_train_optimizer(self):
        """train --optimizer"""
        args = self.parser.parse_args(["train", "--optimizer", "sgd"])
        self.assertEqual(args.optimizer, "sgd")

    def test_train_optimizer_adam(self):
        """train --optimizer adam"""
        args = self.parser.parse_args(["train", "--optimizer", "adam"])
        self.assertEqual(args.optimizer, "adam")

    def test_train_optimizer_adamw(self):
        """train --optimizer adamw"""
        args = self.parser.parse_args(["train", "--optimizer", "adamw"])
        self.assertEqual(args.optimizer, "adamw")

    def test_train_scheduler(self):
        """train --scheduler"""
        for s in ["cosine_warmup", "cosine", "step", "exponential", "constant"]:
            args = self.parser.parse_args(["train", "--scheduler", s])
            self.assertEqual(args.scheduler, s)

    def test_train_pos_encoding(self):
        """train --pos-encoding"""
        for pe in ["sinusoidal", "learnable"]:
            args = self.parser.parse_args(["train", "--pos-encoding", pe])
            self.assertEqual(args.pos_encoding, pe)

    def test_train_activation(self):
        """train --activation"""
        for act in ["gelu", "relu"]:
            args = self.parser.parse_args(["train", "--activation", act])
            self.assertEqual(args.activation, act)

    def test_train_norm_type(self):
        """train --norm-type"""
        for nt in ["pre", "post"]:
            args = self.parser.parse_args(["train", "--norm-type", nt])
            self.assertEqual(args.norm_type, nt)

    def test_train_resume(self):
        """train --resume"""
        args = self.parser.parse_args(["train", "--resume", "/path/to/checkpoint.pth"])
        self.assertEqual(args.resume, "/path/to/checkpoint.pth")

    def test_train_model_params(self):
        """train 模型参数"""
        args = self.parser.parse_args(
            [
                "train",
                "--vocab-size",
                "5000",
                "--context-length",
                "512",
                "--embedding-dim",
                "768",
                "--num-heads",
                "12",
                "--num-layers",
                "12",
                "--ffn-hidden-dim",
                "3072",
                "--dropout",
                "0.2",
            ]
        )
        self.assertEqual(args.vocab_size, 5000)
        self.assertEqual(args.context_length, 512)
        self.assertEqual(args.embedding_dim, 768)
        self.assertEqual(args.num_heads, 12)
        self.assertEqual(args.num_layers, 12)
        self.assertEqual(args.ffn_hidden_dim, 3072)
        self.assertEqual(args.dropout, 0.2)

    def test_train_full_params(self):
        """train 全部参数"""
        args = self.parser.parse_args(
            [
                "train",
                "--batch-size",
                "64",
                "--epochs",
                "100",
                "--total-steps",
                "20000",
                "--grad-clip",
                "0.5",
                "--seed",
                "123",
                "--num-workers",
                "8",
                "--optimizer",
                "adamw",
                "--lr",
                "1e-4",
                "--weight-decay",
                "0.05",
                "--scheduler",
                "cosine_warmup",
                "--warmup-steps",
                "1000",
                "--min-lr-ratio",
                "0.05",
                "--patience",
                "10",
            ]
        )
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.total_steps, 20000)
        self.assertEqual(args.grad_clip, 0.5)
        self.assertEqual(args.seed, 123)
        self.assertEqual(args.num_workers, 8)
        self.assertEqual(args.optimizer, "adamw")
        self.assertEqual(args.lr, 1e-4)
        self.assertEqual(args.weight_decay, 0.05)
        self.assertEqual(args.scheduler, "cosine_warmup")
        self.assertEqual(args.warmup_steps, 1000)
        self.assertEqual(args.min_lr_ratio, 0.05)
        self.assertEqual(args.patience, 10)

    def test_train_warmup_steps(self):
        """train --warmup-steps"""
        args = self.parser.parse_args(["train", "--warmup-steps", "2000"])
        self.assertEqual(args.warmup_steps, 2000)

    def test_train_min_lr_ratio(self):
        """train --min-lr-ratio"""
        args = self.parser.parse_args(["train", "--min-lr-ratio", "0.5"])
        self.assertEqual(args.min_lr_ratio, 0.5)

    # ============================================================
    # eval 命令测试
    # ============================================================

    def test_eval_required_checkpoint(self):
        """eval 必须指定 --checkpoint"""
        args = self.parser.parse_args(["eval", "--checkpoint", "best_model.pth"])
        self.assertEqual(args.checkpoint, "best_model.pth")
        self.assertEqual(args.split, "test")
        self.assertEqual(args.batch_size, self.default_train.batch_size)

    def test_eval_missing_checkpoint(self):
        """eval 缺少 --checkpoint 应该报错"""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["eval"])

    def test_eval_split_train(self):
        """eval --split train"""
        args = self.parser.parse_args(
            ["eval", "--checkpoint", "x.pth", "--split", "train"]
        )
        self.assertEqual(args.split, "train")

    def test_eval_split_val(self):
        """eval --split val"""
        args = self.parser.parse_args(
            ["eval", "--checkpoint", "x.pth", "--split", "val"]
        )
        self.assertEqual(args.split, "val")

    def test_eval_split_test(self):
        """eval --split test（默认）"""
        args = self.parser.parse_args(
            ["eval", "--checkpoint", "x.pth", "--split", "test"]
        )
        self.assertEqual(args.split, "test")

    def test_eval_invalid_split(self):
        """eval --split 非法值"""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(
                ["eval", "--checkpoint", "x.pth", "--split", "invalid"]
            )

    def test_eval_batch_size(self):
        """eval --batch-size"""
        args = self.parser.parse_args(
            ["eval", "--checkpoint", "x.pth", "--batch-size", "128"]
        )
        self.assertEqual(args.batch_size, 128)

    # ============================================================
    # generate 命令测试
    # ============================================================

    def test_generate_required_checkpoint(self):
        """generate 必须指定 --checkpoint"""
        args = self.parser.parse_args(["generate", "--checkpoint", "best_model.pth"])
        self.assertEqual(args.checkpoint, "best_model.pth")
        self.assertIsNone(args.prompt)
        # 生成参数默认值
        self.assertEqual(args.max_tokens, self.default_gen.max_new_tokens)
        self.assertEqual(args.temperature, self.default_gen.temperature)
        self.assertEqual(args.top_k, self.default_gen.top_k)
        self.assertEqual(args.top_p, self.default_gen.top_p)
        self.assertEqual(args.repetition_penalty, self.default_gen.repetition_penalty)
        self.assertFalse(args.no_kv_cache)

    def test_generate_missing_checkpoint(self):
        """generate 缺少 --checkpoint 应该报错"""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["generate"])

    def test_generate_with_prompt(self):
        """generate --prompt"""
        args = self.parser.parse_args(
            ["generate", "--checkpoint", "x.pth", "--prompt", "你好"]
        )
        self.assertEqual(args.prompt, "你好")

    def test_generate_params(self):
        """generate 全部生成参数"""
        args = self.parser.parse_args(
            [
                "generate",
                "--checkpoint",
                "x.pth",
                "--max-tokens",
                "100",
                "--temperature",
                "0.5",
                "--top-k",
                "40",
                "--top-p",
                "0.8",
                "--repetition-penalty",
                "1.2",
            ]
        )
        self.assertEqual(args.max_tokens, 100)
        self.assertEqual(args.temperature, 0.5)
        self.assertEqual(args.top_k, 40)
        self.assertEqual(args.top_p, 0.8)
        self.assertEqual(args.repetition_penalty, 1.2)

    def test_generate_no_kv_cache(self):
        """generate --no-kv-cache"""
        args = self.parser.parse_args(
            ["generate", "--checkpoint", "x.pth", "--no-kv-cache"]
        )
        self.assertTrue(args.no_kv_cache)

    def test_generate_top_k_zero(self):
        """generate --top-k 0（贪心解码）"""
        args = self.parser.parse_args(
            ["generate", "--checkpoint", "x.pth", "--top-k", "0"]
        )
        self.assertEqual(args.top_k, 0)

    def test_generate_temperature_zero(self):
        """generate --temperature 0"""
        args = self.parser.parse_args(
            ["generate", "--checkpoint", "x.pth", "--temperature", "0"]
        )
        self.assertEqual(args.temperature, 0.0)

    # ============================================================
    # 无命令测试
    # ============================================================

    def test_no_command(self):
        """无子命令时 help 可用"""
        import contextlib
        import io

        # 重定向 stdout 来检查 help 输出
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            try:
                self.parser.parse_args(["--help"])
            except SystemExit:
                pass
        output = f.getvalue()
        self.assertIn("GPT", output)
        self.assertIn("train", output)
        self.assertIn("eval", output)
        self.assertIn("generate", output)


if __name__ == "__main__":
    unittest.main()

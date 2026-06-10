"""
src/inference/generator.py

文本生成器
"""

from typing import Optional

import torch
import torch.nn as nn

from config.defaults import GenerationConfig


class TextGenerator:
    """
    文本生成器

    使用方式:
        generator = TextGenerator(model, tokenizer, config)
        text = generator.generate("今天天气")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: GenerationConfig,
        device: str = "cuda",
    ):
        """
        初始化生成器

        Args:
            model: 模型
            tokenizer: 分词器
            config: 生成配置
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        生成文本

        Args:
            prompt: 输入提示文本
            max_new_tokens: 最大生成 token 数
            temperature: 温度
            top_k: Top-K 采样
            top_p: Nucleus 采样

        Returns:
            str: 生成的文本
        """
        # 使用传入参数或默认配置
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = (
            temperature if temperature is not None else self.config.temperature
        )
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p

        # 编码输入
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # 生成循环
        for _ in range(max_new_tokens):
            # 前向传播
            logits = self.model(input_ids)  # (batch_size, seq_len, vocab_size)

            # 取最后一个 token 的 logits
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # 温度缩放
            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Top-K 过滤
            if top_k > 0:
                next_token_logits = self._top_k_filtering(next_token_logits, top_k)

            # Nucleus (Top-P) 过滤
            if top_p < 1.0:
                next_token_logits = self._top_p_filtering(next_token_logits, top_p)

            # 采样或贪心解码
            if temperature > 0:
                # Multinomial 采样
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪心解码 (temperature = 0)
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # 拼接到输入序列
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # 检查是否生成结束符
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # 解码生成的文本
        generated_ids = input_ids[0].tolist()
        generated_text = self.tokenizer.decode(generated_ids)

        return generated_text

    def _top_k_filtering(
        self,
        logits: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """
        Top-K 过滤：保留概率最高的 K 个 token

        Args:
            logits: logits 张量，形状 (batch_size, vocab_size)
            top_k: K 值

        Returns:
            torch.Tensor: 过滤后的 logits
        """
        # 获取 top-k 的值和索引
        top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)

        # 创建 mask，只保留 top-k 的位置
        mask = torch.ones_like(logits) * float("-inf")
        mask.scatter_(-1, top_k_indices, top_k_values)

        return mask

    def _top_p_filtering(
        self,
        logits: torch.Tensor,
        top_p: float,
    ) -> torch.Tensor:
        """
        Nucleus (Top-P) 采样：保留累积概率达到 top_p 的最小 token 集合

        Args:
            logits: logits 张量，形状 (batch_size, vocab_size)
            top_p: 累积概率阈值

        Returns:
            torch.Tensor: 过滤后的 logits
        """
        # 按概率从大到小排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)

        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 找到累积概率超过 top_p 的位置
        sorted_indices_to_remove = cumulative_probs > top_p

        # 保留第一个超过 top_p 的 token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # 创建 mask
        mask = torch.ones_like(logits) * float("-inf")
        mask.scatter_(-1, sorted_indices, sorted_logits)

        # 应用 mask
        mask[
            sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
        ] = float("-inf")

        return mask


if __name__ == "__main__":
    print("=" * 60)
    print("测试文本生成器")
    print("=" * 60)

    import torch.nn as nn

    # 创建简单测试模型
    class DummyModel(nn.Module):
        def __init__(self, vocab_size=1000):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 128)
            self.linear = nn.Linear(128, vocab_size)

        def forward(self, input_ids):
            x = self.embedding(input_ids)
            logits = self.linear(x)
            return logits

    # 创建简单测试 tokenizer
    class DummyTokenizer:
        def __init__(self):
            self.eos_token_id = 2

        def encode(self, text):
            return [1, 2, 3]  # 简单编码

        def decode(self, token_ids):
            return "生成的文本"  # 简单解码

    model = DummyModel()
    tokenizer = DummyTokenizer()
    config = GenerationConfig(
        max_new_tokens=10,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        do_sample=True,
    )

    generator = TextGenerator(model, tokenizer, config, device="cpu")

    print("\n生成器创建成功")
    print(
        f"配置: max_new_tokens={config.max_new_tokens}, temperature={config.temperature}"
    )
    print(f"      top_k={config.top_k}, top_p={config.top_p}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

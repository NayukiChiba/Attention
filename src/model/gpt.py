"""
src/model/gpt.py

GPT 模型实现

"""

import torch
import torch.nn as nn

from config.defaults import GPTConfig
from src.model.embedding import GPTEmbedding
from src.model.transformerBlock import TransformerBlock


class GPT(nn.Module):
    """
    GPT 语言模型

    结构:
        1. Token Embedding + Positional Embedding
        2. N 层 Transformer Decoder Block
        3. Layer Normalization, 仅Pre-Norm
        4. Langauge Modeling Head (线性层), 输出 logits

    Args:
        config: GPTConfig 对象,包含模型超参数


    """

    def __init__(self, config: GPTConfig):
        super(GPT, self).__init__()
        self.config = config

        # 1. Token Embedding + Positional Embedding
        self.embedding = GPTEmbedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            block_size=config.context_length,
            dropout_rate=config.dropout_rate,
            pos_encoding_type=config.pos_encoding_type,
        )

        # 2. N 层 Transformer Decoder Block
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=config.embedding_dim,
                    num_attention_heads=config.num_attention_heads,
                    ffn_hidden_dim=config.ffn_hidden_dim,
                    dropout_rate=config.dropout_rate,
                    activation=config.activation,
                    norm_type=config.norm_type,
                    layer_norm_eps=config.layer_norm_eps,
                )
                for _ in range(config.num_layers)
            ]
        )

        # 3. Layer Normalization, 仅Pre-Norm
        if config.norm_type == "pre":
            self.final_norm = nn.LayerNorm(
                config.embedding_dim, eps=config.layer_norm_eps
            )
        else:
            self.final_norm = nn.Identity()  # 如果不是 Pre-Norm,则不使用 LayerNorm

        # 4. Language Modeling Head (线性层), 输出 logits
        self.language_modeling_head = nn.Linear(
            config.embedding_dim, config.vocab_size, bias=False
        )

        # 权重共享
        if config.share_embedding_weights:
            # 让输出层和 token embedding 层共享权重
            self.language_modeling_head.weight = (
                self.embedding.token_embedding.embedding.weight
            )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        初始化模型权重
        Args:
            module: 模块对象


        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        """
        前向传播
        Args:
            input_ids: 输入 token ID 张量,形状为 (batch_size, seq_length)
            attention_mask: 可选的注意力掩码张量,形状为 (batch_size, seq_length, seq_length),True 表示需要 mask
            return_attention_weights: 是否返回每层的注意力权重
        Returns:
            torch.Tensor: 输出 logits 张量,形状为 (batch_size, seq_length, vocab_size)
            如果 return_attention_weights=True，返回 (logits, list_of_attn_weights)
        """
        # 1. Embedding
        # (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        x = self.embedding(input_ids)

        # 2. Transformer Blocks
        all_attn_weights = []
        for block in self.transformer_blocks:
            result = block(
                x, attention_mask, return_attention_weights=return_attention_weights
            )
            if return_attention_weights:
                x, attn_weights = result
                all_attn_weights.append(attn_weights)
            else:
                x = result

        # 3. Final Layer Normalization (仅 Pre-Norm)
        x = self.final_norm(x)

        # 4. Language Modeling Head
        logits = self.language_modeling_head(x)

        if return_attention_weights:
            return logits, all_attn_weights
        return logits

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        获取模型参数量

        Args:
            non_embedding (bool): 是否排除 embedding 层参数，默认 False

        Returns:
            int: 参数量
        """
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            # 排除 token embedding 和 positional embedding
            n_params -= self.embedding.token_embedding.embedding.weight.numel()
            # 如果位置编码是可学习的,也要排除它的参数
            if self.config.pos_encoding_type == "learnable":
                n_params -= self.embedding.pos_embedding.weight.numel()

        return n_params


if __name__ == "__main__":
    print("=" * 60)
    print("测试 GPT 模型")
    print("=" * 60)

    from config.defaults import GPTConfig

    # 创建配置（小模型用于测试）
    config = GPTConfig(
        vocab_size=5000,
        context_length=256,
        embedding_dim=384,
        num_attention_heads=6,
        num_layers=6,
        ffn_hidden_dim=1536,
        dropout_rate=0.1,
        pos_encoding_type="learnable",
        activation="gelu",
        norm_type="pre",
        layer_norm_eps=1e-5,
        share_embedding_weights=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}\n")

    print("模型配置:")
    config._summary()
    print()

    # 1. 创建模型
    print("1. 创建模型")
    print("-" * 60)

    model = GPT(config).to(device)

    total_params = model.get_num_params()
    non_embedding_params = model.get_num_params(non_embedding=True)

    print(f"总参数量: {total_params:,}")
    print(f"非 Embedding 参数量: {non_embedding_params:,}")
    print(f"Embedding 参数量: {total_params - non_embedding_params:,}")
    print()

    # 2. 测试前向传播
    print("2. 测试前向传播")
    print("-" * 60)

    batch_size = 4
    seq_len = 128

    # 创建随机 token ids
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )
    print(f"Input shape: {input_ids.shape}")

    # 前向传播
    logits = model(input_ids)
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "输出形状错误"
    print("✓ 前向传播测试通过\n")

    # 3. 测试带 mask 的前向传播
    print("3. 测试带 mask 的前向传播")
    print("-" * 60)

    from src.model.mask import create_causal_mask

    causal_mask = create_causal_mask(seq_len, device=device)
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    print(f"Causal mask shape: {causal_mask.shape}")

    logits_with_mask = model(input_ids, attention_mask=causal_mask)
    print(f"Output logits with mask shape: {logits_with_mask.shape}")
    assert logits_with_mask.shape == (batch_size, seq_len, config.vocab_size), (
        "带mask的输出形状错误"
    )
    print("✓ 带 mask 的前向传播测试通过\n")

    # 4. 测试梯度反向传播
    print("4. 测试梯度反向传播")
    print("-" * 60)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 创建输入和目标
    input_ids_grad = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )
    target_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )

    # 前向传播
    logits_grad = model(input_ids_grad, attention_mask=causal_mask)

    # 计算损失（交叉熵）
    loss = torch.nn.functional.cross_entropy(
        logits_grad.view(-1, config.vocab_size),
        target_ids.view(-1),
    )
    print(f"Loss: {loss.item():.4f}")

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 检查梯度
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params_count = sum(1 for _ in model.parameters())
    print(f"有梯度的参数: {has_grad}/{total_params_count}")
    assert has_grad == total_params_count, "部分参数没有梯度"

    # 更新参数
    optimizer.step()
    print("✓ 梯度反向传播测试通过\n")

    # 5. 测试权重共享
    print("5. 测试权重共享")
    print("-" * 60)

    if config.share_embedding_weights:
        embedding_weight = model.embedding.token_embedding.embedding.weight
        language_modeling_weight = model.language_modeling_head.weight

        print(f"Embedding weight shape: {embedding_weight.shape}")
        print(f"Language modeling head weight shape: {language_modeling_weight.shape}")
        print(
            f"权重是否共享: {embedding_weight.data_ptr() == language_modeling_weight.data_ptr()}"
        )
        assert embedding_weight.data_ptr() == language_modeling_weight.data_ptr(), (
            "权重没有共享"
        )
        print("✓ 权重共享测试通过\n")
    else:
        print("权重共享未启用\n")

    # 6. 测试不同的 norm_type
    print("6. 测试不同的 norm_type")
    print("-" * 60)

    for norm_type in ["pre", "post"]:
        config_norm = GPTConfig(
            vocab_size=1000,
            context_length=64,
            embedding_dim=128,
            num_attention_heads=4,
            num_layers=2,
            ffn_hidden_dim=512,
            norm_type=norm_type,
        )

        model_norm = GPT(config_norm).to(device)
        input_norm = torch.randint(0, 1000, (2, 64), device=device)
        output_norm = model_norm(input_norm)

        print(
            f"norm_type={norm_type}: 输出形状={output_norm.shape}, 参数量={model_norm.get_num_params():,}"
        )

    print("✓ 不同 norm_type 测试通过\n")

    # 7. 测试不同的 pos_encoding_type
    print("7. 测试不同的 pos_encoding_type")
    print("-" * 60)

    for pos_type in ["sinusoidal", "learnable"]:
        config_pos = GPTConfig(
            vocab_size=1000,
            context_length=64,
            embedding_dim=128,
            num_attention_heads=4,
            num_layers=2,
            ffn_hidden_dim=512,
            pos_encoding_type=pos_type,
        )

        model_pos = GPT(config_pos).to(device)
        input_pos = torch.randint(0, 1000, (2, 64), device=device)
        output_pos = model_pos(input_pos)

        print(
            f"pos_encoding_type={pos_type}: 输出形状={output_pos.shape}, 参数量={model_pos.get_num_params():,}"
        )

    print("✓ 不同 pos_encoding_type 测试通过\n")

    print("=" * 60)
    print("所有测试通过！")
    print("=" * 60)

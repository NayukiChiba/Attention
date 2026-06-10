"""
src/model/gpt.py

GPT 模型实现

"""

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
        config: GPTConfig 对象，包含模型超参数


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
                    feed_forward_dim=config.ffn_hidden_dim,
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
            self.final_norm = nn.Identity()  # 如果不是 Pre-Norm，则不使用 LayerNorm

        # 4. Language Modeling Head (线性层), 输出 logits
        self.language_modeling_head = nn.Linear(
            config.embedding_dim, config.vocab_size, bias=False
        )

        # 权重共享
        if config.share_embedding_weights:
            # 让输出层和 token embedding 层共享权重
            self.language_modeling_head.weight = self.embedding.token_embedding.weight

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        pass

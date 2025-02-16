from typing import Any, Optional, Tuple
from functools import partial
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax
from rich import print
from tqdm import tqdm


class Qwen2Config:
    def __init__(
        self,
        vocab_size=151936,
        hidden_size=1536,
        intermediate_size=8960,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout


class Qwen2RMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.weight = self.param(
            "weight",
            nn.initializers.ones,
            (self.hidden_size,),
            self.param_dtype,
        )

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * lax.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).astype(input_dtype)


class Qwen2MLP(nn.Module):
    config: Qwen2Config
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.gate_proj = nn.Dense(
            features=self.config.intermediate_size,
            use_bias=False,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )
        self.up_proj = nn.Dense(
            features=self.config.intermediate_size,
            use_bias=False,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )
        self.down_proj = nn.Dense(
            features=self.config.hidden_size,
            use_bias=False,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Attention(nn.Module):
    config: Qwen2Config
    layer_idx: int
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.num_key_value_groups = (
            self.config.num_attention_heads // self.config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Dense(
            features=self.config.num_attention_heads * self.head_dim,
            use_bias=True,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )
        self.k_proj = nn.Dense(
            features=self.config.num_key_value_heads * self.head_dim,
            use_bias=True,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )
        self.v_proj = nn.Dense(
            features=self.config.num_key_value_heads * self.head_dim,
            use_bias=True,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )
        self.o_proj = nn.Dense(
            features=self.config.hidden_size,
            use_bias=False,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):
        cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
        sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def repeat_kv(self, hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
        if n_rep == 1:
            return hidden_states
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = jnp.expand_dims(hidden_states, axis=2)
        hidden_states = jnp.broadcast_to(
            hidden_states, (batch, num_key_value_heads, n_rep, slen, head_dim)
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        position_embeddings: Tuple[jnp.ndarray, jnp.ndarray],
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        batch_size, seq_length = hidden_states.shape[:2]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(
            batch_size, seq_length, self.config.num_attention_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            batch_size, seq_length, self.config.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            batch_size, seq_length, self.config.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        # Scaled dot-product attention
        attn_weights = (
            jnp.matmul(query_states, key_states.transpose(0, 1, 3, 2)) * self.scaling
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.softmax(attn_weights, axis=-1)

        if not deterministic and self.config.attention_dropout > 0:
            attn_weights = nn.dropout(
                attn_weights,
                rate=self.config.attention_dropout,
                deterministic=deterministic,
            )

        attn_output = jnp.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_length, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None


class Qwen2DecoderLayer(nn.Module):
    config: Qwen2Config
    layer_idx: int
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self_attn = Qwen2Attention(
            config=self.config,
            layer_idx=self.layer_idx,
            param_dtype=self.param_dtype,
        )
        self.mlp = Qwen2MLP(
            config=self.config,
            param_dtype=self.param_dtype,
        )
        self.input_layernorm = Qwen2RMSNorm(
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            param_dtype=self.param_dtype,
        )
        self.post_attention_layernorm = Qwen2RMSNorm(
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        position_embeddings: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen2Model(nn.Module):
    config: Qwen2Config
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_tokens = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )

        self.layers = [
            Qwen2DecoderLayer(
                config=self.config,
                layer_idx=i,
                param_dtype=self.param_dtype,
            )
            for i in range(self.config.num_hidden_layers)
        ]

        self.norm = Qwen2RMSNorm(
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            param_dtype=self.param_dtype,
        )

    def _compute_rope_embeddings(self, hidden_states, position_ids, inv_freq=None):
        if inv_freq is None:
            dim = self.config.hidden_size // self.config.num_attention_heads
            inv_freq = 1.0 / (
                self.config.rope_theta
                ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
            )

        inv_freq = jnp.expand_dims(jnp.expand_dims(inv_freq, 0), 0)
        position_ids = jnp.expand_dims(position_ids.astype(jnp.float32), -1)
        freqs = jnp.matmul(position_ids, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)

        return cos, sin

    def _prepare_causal_attention_mask(
        self,
        attention_mask: jnp.ndarray,
        sequence_length: int,
        dtype: jnp.dtype,
        batch_size: int,
    ):
        min_dtype = jnp.finfo(dtype).min
        causal_mask = jnp.triu(
            jnp.full((sequence_length, sequence_length), min_dtype, dtype=dtype),
            k=1,
        )
        causal_mask = jnp.broadcast_to(
            causal_mask[None, None, :, :],
            (batch_size, 1, sequence_length, sequence_length),
        )

        if attention_mask is not None:
            causal_mask = causal_mask + attention_mask

        return causal_mask

    def __call__(
        self,
        input_ids: jnp.ndarray = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = jnp.arange(inputs_embeds.shape[1])[None, :]

        # Create causal mask
        causal_mask = self._prepare_causal_attention_mask(
            attention_mask,
            inputs_embeds.shape[1],
            inputs_embeds.dtype,
            inputs_embeds.shape[0],
        )

        hidden_states = inputs_embeds

        # Create position embeddings to be shared across layers
        position_embeddings = self._compute_rope_embeddings(hidden_states, position_ids)

        # Process through layers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs += (all_hidden_states,)
        if output_attentions:
            outputs += (all_attentions,)

        return outputs


class Qwen2ForCausalLM(nn.Module):
    config: Qwen2Config
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.model = Qwen2Model(
            config=self.config,
            param_dtype=self.param_dtype,
        )
        self.lm_head = nn.Dense(
            features=self.config.vocab_size,
            use_bias=False,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        labels: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # Loss computation
        loss = None
        if labels is not None:
            vocab_size = logits.shape[-1]
            # Shift logits and labels
            shift_logits = logits[..., :-1, :].reshape(-1, vocab_size)
            shift_labels = labels[..., 1:].reshape(-1)

            one_hot = jax.nn.one_hot(shift_labels, vocab_size)
            mask = shift_labels != -1  # Assuming -1 is the padding token
            losses = -jnp.sum(one_hot * nn.log_softmax(shift_logits, axis=-1), axis=-1)
            loss = jnp.sum(losses * mask) / jnp.sum(mask)

        return (loss, logits) + outputs[1:]

    def generate(
        self,
        input_ids: jnp.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        prng_key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """Generate tokens autoregressively without using KV cache."""
        generated = input_ids
        batch_size = generated.shape[0]

        for _ in tqdm(range(max_new_tokens)):
            # Model forward pass with full sequence
            outputs = self(
                input_ids=generated,
                deterministic=True,
            )
            logits = outputs[1]

            # Get logits for last token and scale by temperature
            next_token_logits = logits[:, -1, :] / temperature

            # Optionally apply top-k
            if top_k is not None:
                top_logits, _ = jax.lax.top_k(next_token_logits, top_k)
                k_threshold = jnp.min(top_logits, axis=-1, keepdims=True)
                next_token_logits = jnp.where(
                    next_token_logits < k_threshold, -jnp.inf, next_token_logits
                )

            # Convert to probabilities
            probs = jax.nn.softmax(next_token_logits, axis=-1)

            # Sample or take argmax
            if do_sample and prng_key is not None:
                prng_key, subkey = jax.random.split(prng_key)
                next_token = jax.random.categorical(subkey, next_token_logits, axis=-1)
            else:
                next_token = jnp.argmax(probs, axis=-1)

            # Append generated token
            next_token = next_token[:, None]  # Add sequence dimension
            generated = jnp.concatenate([generated, next_token], axis=1)

        return generated

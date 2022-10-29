import random
import torch
from transformers import AutoTokenizer
from typing import Sequence, Tuple

EXAMPLE_Text = ["best hotel in bay area", "here is an example of gpt2 model"]


def get_tokenizer(model_name_or_path: str, cache_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_example_inputs(
    model_name_or_path: str,
    cache_dir: str,
    num_attention_heads: int,
    num_layer: int,
    hidden_size: int,
    device: str,
    prompt_text: Sequence[str] = EXAMPLE_Text,
):
    tokenizer = get_tokenizer(model_name_or_path, cache_dir)
    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)

    input_ids = torch.tensor(encodings_dict["input_ids"], dtype=torch.int32)
    attention_mask = torch.tensor(encodings_dict["attention_mask"], dtype=torch.int32)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(position_ids < 0, 0)
    position_ids = position_ids.to(torch.int32)

    # Empty Past State for generating first word
    empty_past = []
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    past_shape = [
        2,
        batch_size,
        num_attention_heads,
        0,
        hidden_size // num_attention_heads,
    ]
    for i in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))

    return input_ids, attention_mask, position_ids, empty_past


def get_dummy_inputs(
    batch_size: int,
    past_sequence_length: int,
    sequence_length: int,
    num_attention_heads: int,
    hidden_size: int,
    num_layer: int,
    vocab_size: int,
    device: torch.device,
    has_position_ids: bool = True,
    has_attention_mask: bool = True,
    input_ids_dtype: torch.dtype = torch.int64,
    position_ids_dtype: torch.dtype = torch.int64,
    attention_mask_dtype: torch.dtype = torch.int64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random inputs for GPT2 model.
    Returns torch tensors of input_ids, position_ids, attention_mask and a list of past state tensors.
    """
    past_shape = [
        2,
        batch_size,
        num_attention_heads,
        past_sequence_length,
        int(hidden_size / num_attention_heads),
    ]

    past = [
        (torch.rand(past_shape, dtype=torch.float32, device=device) * 2.0 - 1.0)
        for _ in range(num_layer)
    ]
    input_ids = torch.randint(
        low=0,
        high=vocab_size - 1,
        size=(batch_size, sequence_length),
        dtype=input_ids_dtype,
        device=device,
    )

    attention_mask = None
    if has_attention_mask:
        total_sequence_length = past_sequence_length + sequence_length
        attention_mask = torch.ones(
            [batch_size, total_sequence_length],
            dtype=attention_mask_dtype,
            device=device,
        )
        if total_sequence_length >= 2:
            padding_position = random.randint(
                0, total_sequence_length - 1
            )  # test input with padding.
            attention_mask[:, padding_position] = 0

    # Deduce position_ids from attention mask
    position_ids = None
    if has_position_ids:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(position_ids < 0, 0)
        position_ids = position_ids[:, past_sequence_length:].to(position_ids_dtype)

    return (input_ids, attention_mask, position_ids, past)

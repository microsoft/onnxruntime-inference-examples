import torch
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase, AutoConfig
from typing import List, Optional, Union
from optimum.onnxruntime import ORTModelForCausalLM

from lm_eval import utils
from lm_eval.base import BaseLM

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]


class ORTCausalLM(BaseLM):
    _DEFAULT_MAX_LENGTH: int = 2048

    def __init__(self,
                 pretrained: str,
                 tokenizer: Optional[str] = None,
                 subfolder: Optional[str] = None,
                 revision: Optional[str] = "main",
                 batch_size: Optional[Union[int, str]] = 1,
                 max_gen_toks: Optional[int] = 1024,
                 max_length: Optional[int] = None,
                 add_special_tokens: Optional[bool] = None,
                 device: Optional[Union[int, str]] = "cpu",
                 load_in_8bit: Optional[bool] = False,
                 trust_remote_code: Optional[bool] = False,
                 **kwargs,
                 ):
        """Initializes an ORT `CausalModel` and huggingface `AutoTokenizer` for evaluation.
        Args:
            pretrained (str):
                The Path to the ONNX model to be loaded. This is effectively the 
                `pretrained_model_name_or_path` argument of `from_pretrained` in the 
                optimum `onnxruntime` API.
            add_special_tokens (bool, optional, defaults to True):
                Whether to add special tokens to the input sequences. If `None`, the
                default value will be set to `True` for seq2seq models (e.g. T5) and
                `False` for causal models.
                WARNING: Evaluating causal models with `add_special_tokens=True` is
                currently __not__ supported.
            load_in_8bit (bool, optional, defaults to False):
                If True, will convert the loaded model into mixed-8bit quantized model. See:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.load_in_8bit
            trust_remote_code (bool, optional, defaults to False):
                If True, will trust the remote code when loading the model.
        """
        super().__init__()

        assert isinstance(pretrained, str)
        assert isinstance(device, str)
        assert isinstance(batch_size, (int, str))
        if (
                add_special_tokens is not None
        ):
            # TODO: Support evaluating causal models with special tokens. Currently,
            #  this is not possible because the `_loglikelihood_tokens()` method for
            #  causal LMs makes a no-special-tokens assumption given that contexts
            #  and labels/continuations are tokenized separately without special
            #  tokens, concatenated, and then processed as inputs.
            assert (
                not add_special_tokens
            ), "Evaluating causal models with `add_special_tokens=True` is currently not supported."

        device_list = set(
            ["cuda", "cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        )
        if device and device in device_list:
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        # setup for automatic batch size detection
        if batch_size == "auto":
            self._batch_size = batch_size
        else:
            self._batch_size = int(batch_size)

        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self._add_special_tokens = add_special_tokens
        model_kwargs = {"load_in_8bit": load_in_8bit}
        self._config = AutoConfig.from_pretrained(pretrained)
        print(f"Executing ONNX models under {pretrained}")
        self.model: ORTModelForCausalLM = ORTModelForCausalLM.from_pretrained(
            pretrained,
            revision=revision,
            subfolder='' if subfolder is None else subfolder,
            trust_remote_code=trust_remote_code,
            config=self._config,
            **kwargs,
            **model_kwargs,
        )
        try:
            self.tokenizer: PreTrainedTokenizerBase = self.model.preprocessors[0]
        except IndexError:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained if tokenizer is None else tokenizer,
                revision=revision + ("/" + subfolder if subfolder is not None else ""),
            )
        self.tokenizer.model_max_length = self.max_length
        self._padding = self.tokenizer.pad_token is not None

        torch.set_grad_enabled(False)

    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
         check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        else:
            return False

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.
        """
        if self._max_length is not None:
            return self._max_length
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx", "max_sequence_length")
        for attr in seqlen_config_attrs:
            if self._config and hasattr(self._config, attr):
                return getattr(self._config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def batch_size(self) -> int:
        # TODO: Add adaptive batch size.
        return self._batch_size  # * gpus

    @property
    def device(self) -> Union[int, str, torch.device]:
        return self._device

    def _model_call(self, inputs: TokenSequence) -> TokenSequence:
        attention_mask = torch.ones(inputs.shape, dtype=torch.int64, device=self._device)
        with torch.no_grad():
            logits = self.model(inputs, attention_mask)["logits"]
            return logits

    def _model_generate(
            self,
            inputs: BatchEncoding,
            max_tokens: int,
            eos_token_id: int,
    ) -> TokenSequence:
        # Ensure that the context does not encroach into the `space`
        # for the generation.
        inputs["input_ids"] = inputs["input_ids"][:, self.max_gen_toks - self.max_length:]
        inputs["attention_mask"] = inputs["attention_mask"][
                                   :, self.max_gen_toks - self.max_length:
                                   ]

        generation_kwargs = {"do_sample": False, "max_length": max_tokens}
        if eos_token_id is not None:
            generation_kwargs['eos_token_id'] = eos_token_id
            generation_kwargs['pad_token_id'] = eos_token_id  # setting eos_token_id as pad token

        generations = self.model.generate(
            **inputs,
            **generation_kwargs,
        )
        return utils.select_continuation_from_batch_left_padding(
            generations, max_context_size=inputs["input_ids"].size(1)
        )

    def tok_encode(self, string: str) -> TokenSequence:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings: Union[str, List[str], List[List[str]]]) -> TokenSequence:
        inputs = self.tokenizer(
            strings,
            padding=self._padding,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        return inputs

    def tok_decode(self, tokens: torch.LongTensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

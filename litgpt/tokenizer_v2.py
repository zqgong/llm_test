from transformers import AutoTokenizer
import json
from pathlib import Path
from typing import Optional, Union

import torch


class TTokenizer:
    def __init__(self, checkpoint_dir: Union[Path, str], special_tokens=None) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise NotADirectoryError(f"The checkpoint directory does not exist: {str(checkpoint_dir)}")

        self.use_bos = self.check_if_bos_token_used(checkpoint_dir)
        
        self.processor = build_tokenizer(checkpoint_dir, special_tokens, False)

        self.bos_id = self.processor.bos_token_id
        self.eos_id = self.processor.eos_token_id
        self.pad_token_id = self.processor.pad_token_id


    @property
    def vocab_size(self) -> int:
        return len(self.processor)

    def token_to_id(self, token: str) -> int:
        return self.processor.convert_tokens_to_ids([token])[0]

    def check_if_bos_token_used(self, checkpoint_dir: Path) -> bool:
        if not (tokenizer_config_path := checkpoint_dir / "tokenizer_config.json").is_file():
            return False
        with open(tokenizer_config_path, encoding="utf-8") as fp:
            config = json.load(fp)
        if "add_bos_token" in config:
            return config["add_bos_token"]
        # if `add_bos_token` isn't in the config file, but LLaMA tokenizer is used - return True.
        # ex: https://huggingface.co/stabilityai/StableBeluga2/blob/main/tokenizer_config.json#L2
        return config.get("tokenizer_class") == "LlamaTokenizer"

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: Optional[bool] = None,
        eos: bool = False,
        max_length: int = -1,
        return_mask: bool = False,
        return_tensors: str = "pt",
        add_special_tokens: bool = False,
    ) -> torch.Tensor:

        tokens = self.processor(string, add_special_tokens=add_special_tokens)
        if bos or (bos is None and self.use_bos):
            bos_id = self.bos_id
            if bos_id is None:
                raise NotImplementedError("This tokenizer does not have a defined a bos token")
            tokens["input_ids"] = [bos_id] + tokens["input_ids"]
            tokens["attention_mask"] = [1] + tokens["attention_mask"]
        if eos:
            tokens["input_ids"] = tokens["input_ids"] + [self.eos_id]
            tokens["attention_mask"] = tokens["attention_mask"] + [1]
        if max_length > 0:
            tokens["input_ids"] = tokens["input_ids"][:max_length]
            tokens["attention_mask"] = tokens["attention_mask"][:max_length]

        if return_mask:
            if return_tensors == "pt":
                tokens["input_ids"] = torch.tensor([tokens["input_ids"]], dtype=torch.int, device=device)
                tokens["attention_mask"] = torch.tensor([tokens["attention_mask"]], dtype=torch.int, device=device)
            return tokens

        return torch.tensor(tokens["input_ids"], dtype=torch.int, device=device) if return_tensors == "pt" else tokens["input_ids"]

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        return self.processor.decode(tokens)

    def pad(self, inp, return_tensors="pt", padding="longest"):
        return self.processor.pad(
            inp, return_tensors=return_tensors,
        )


def infer_name(name):
    KNOWN_NAMES = ["qwen", "llama", "vicuna", "baichuan", "internlm", "stablelm", "chatglm"]

    for i, k in enumerate(KNOWN_NAMES):
        if k in name:
            return KNOWN_NAMES[i]

    print("ERROR: maybe not yet support")

    return name.split("/")[-1]


def build_tokenizer(tokenizer_path: str, special_tokens=["<|endofchunk|>", "<image>"], add_bos_token=True):
    all_special_tokens = None
    if special_tokens is not None and len(special_tokens) > 0:
        all_special_tokens = {"additional_special_tokens": special_tokens} # ["<|endofchunk|>", "<image>"]
    
    lower_tokenizer_path = str(tokenizer_path).lower()
    name = infer_name(lower_tokenizer_path)
    print("name", name)
    if name == "qwen":
        text_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            add_bos_token=add_bos_token,
            # extra_vocab_file="/home/notebook/code/personal/80234819/lmm/models/qwen/qwen_extra.tiktoken"
        )
        print("all_special_tokens", text_tokenizer.all_special_tokens)
        print(len(text_tokenizer)) # no extra 151851.  

        if text_tokenizer.bos_token_id is None:
            text_tokenizer.bos_token = '<|extra_0|>'
        if text_tokenizer.eos_token_id is None:
            text_tokenizer.eos_token = '<|endoftext|>'
        if text_tokenizer.pad_token_id is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token

    else:
        text_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            add_bos_token=add_bos_token,
        )

        if all_special_tokens is not None and len(all_special_tokens) > 0:
            text_tokenizer.add_special_tokens(all_special_tokens)
        print("all_special_tokens", text_tokenizer.all_special_tokens)

        if name == "chatglm":
            text_tokenizer.bos_token_id = [64790, 64792]

        if name == "stablelm":
            text_tokenizer.pad_token = '<|endoftext|>'
            text_tokenizer.bos_token = '<s>'
            text_tokenizer.eos_token = '</s>'

        if text_tokenizer.pad_token is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token

    print("text_tokenizer bos", text_tokenizer.bos_token_id, "eos", text_tokenizer.eos_token_id, "pad", text_tokenizer.pad_token_id, "len", len(text_tokenizer))

    return text_tokenizer
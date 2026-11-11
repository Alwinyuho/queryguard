"""Language model backends: a local GPU model, or any OpenAI-compatible API."""

from __future__ import annotations

import os
from typing import Protocol

import httpx

from . import config

Messages = list[dict[str, str]]


class LLM(Protocol):
    name: str

    def generate_batch(self, batch: list[Messages]) -> list[str]: ...


class LocalLLM:
    """A small open model on the GPU, in half precision or 4-bit, generating in batches.

    On an 8 GB laptop GPU, a 3B model in half precision was faster than a 4B model
    in 4-bit, because bitsandbytes 4-bit kernels dequantize weights at every step.
    """

    def __init__(
        self,
        model_name: str = config.LOCAL_LLM_MODEL,
        load_in_4bit: bool = config.LOCAL_LLM_4BIT,
        max_batch_tokens: int = config.LOCAL_LLM_BATCH_TOKENS,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"local:{model_name}{' (4-bit)' if load_in_4bit else ''}"
        self.max_batch_tokens = max_batch_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        kwargs: dict = {"device_map": {"": 0}, "dtype": torch.float16}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    def _prompt(self, messages: Messages) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def generate_batch(self, batch: list[Messages], max_new_tokens: int = 400) -> list[str]:
        import torch

        prompts = [self._prompt(m) for m in batch]
        lengths = [len(self.tokenizer(p).input_ids) for p in prompts]
        order = sorted(range(len(prompts)), key=lengths.__getitem__)
        outputs: dict[int, str] = {}

        group: list[int] = []
        groups = []
        for i in order:
            longest = max([lengths[j] for j in group] + [lengths[i]]) + max_new_tokens
            if group and longest * (len(group) + 1) > self.max_batch_tokens:
                groups.append(group)
                group = []
            group.append(i)
        if group:
            groups.append(group)

        for group in groups:
            inputs = self.tokenizer(
                [prompts[i] for i in group], return_tensors="pt", padding=True
            ).to(self.model.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            for i, text in zip(
                group,
                self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True),
                strict=True,
            ):
                outputs[i] = text
        return [outputs[i] for i in range(len(prompts))]


class ApiLLM:
    """Works with any /chat/completions endpoint, such as Groq, Gemini or Ollama."""

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 120.0):
        self.name = f"api:{model}"
        self.model = model
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=timeout,
        )

    def generate_batch(self, batch: list[Messages]) -> list[str]:
        replies = []
        for messages in batch:
            response = self._client.post(
                "/chat/completions",
                json={"model": self.model, "messages": messages, "temperature": 0},
            )
            response.raise_for_status()
            replies.append(response.json()["choices"][0]["message"]["content"] or "")
        return replies


def llm_from_env() -> LLM:
    base_url, model = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_MODEL")
    if base_url and model:
        return ApiLLM(base_url, model, os.environ.get("LLM_API_KEY", ""))
    return LocalLLM()

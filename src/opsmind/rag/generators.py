"""Text-generation adapters used by the RAG service."""

from __future__ import annotations

from typing import Protocol

import torch

from opsmind.exceptions import GenerationModelError


class TextGenerator(Protocol):
    """Minimal provider contract used by the RAG service."""

    @property
    def provider(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def generate(self, prompt: str) -> str: ...


class ExtractiveGenerator:
    """Deterministic offline fallback used for smoke tests and no-model environments."""

    def __init__(self, *, max_chars: int = 900) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than zero.")
        self._max_chars = max_chars

    @property
    def provider(self) -> str:
        return "extractive"

    @property
    def model_name(self) -> str:
        return "deterministic-context-extractor"

    def generate(self, prompt: str) -> str:
        marker = "Context:\n"
        answer_marker = "\n\nAnswer:"
        if marker not in prompt:
            return prompt[: self._max_chars].strip()
        context = prompt.split(marker, 1)[1]
        if answer_marker in context:
            context = context.split(answer_marker, 1)[0]
        text = " ".join(context.split())
        if not text:
            return "The available knowledge base is insufficient."
        return text[: self._max_chars].strip()


class HuggingFaceLocalGenerator:
    """Small local seq2seq model for fully local RAG generation."""

    def __init__(
        self,
        model_name: str,
        *,
        max_new_tokens: int,
        max_input_tokens: int = 512,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be greater than zero.")

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise GenerationModelError(
                "transformers is not installed. Run `python -m pip install -e \".[dev]\"`."
            ) from exc

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._max_input_tokens = max_input_tokens

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self._device)
            self._model.eval()
        except Exception as exc:
            raise GenerationModelError(
                f"Could not load local generation model: {model_name!r}."
            ) from exc

    @property
    def provider(self) -> str:
        return "hf_local"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        try:
            encoded = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_input_tokens,
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.inference_mode():
                output_ids = self._model.generate(
                    **encoded,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                )
            text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        except Exception as exc:
            raise GenerationModelError("Local Hugging Face generation failed.") from exc

        return text or "The available knowledge base is insufficient."


class OpenAICompatibleGenerator:
    """Provider adapter for OpenAI-compatible chat-completions HTTP APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        max_new_tokens: int,
        timeout_seconds: float,
    ) -> None:
        if not api_key.strip():
            raise GenerationModelError(
                "OPSMIND_OPENAI_API_KEY is required when OPSMIND_LLM_PROVIDER=openai_compatible."
            )
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._timeout_seconds = timeout_seconds

    @property
    def provider(self) -> str:
        return "openai_compatible"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        try:
            import httpx
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model_name,
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": self._max_new_tokens,
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            text = str(data["choices"][0]["message"]["content"]).strip()
        except Exception as exc:
            raise GenerationModelError("OpenAI-compatible generation failed.") from exc

        return text or "The available knowledge base is insufficient."

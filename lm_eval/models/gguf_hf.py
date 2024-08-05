import logging
import time
from typing import Dict, List

import requests
from requests.exceptions import RequestException
from tqdm import tqdm

from lm_eval.api.registry import register_model
from lm_eval.models.gguf import GGUFLM

from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


@register_model("gguf_hf", "ggml_hf")
class GGUFHFLM(GGUFLM):
    def __init__(
        self, base_url=None, max_length=2048, tokenizer_name: str = None, **kwargs
    ):
        super().__init__(base_url, max_length, **kwargs)
        assert tokenizer_name, "Must specify tokenizer name"
        self.model_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.kwargs_mapping = {"max_gen_toks": "max_tokens"}

    def gguf_completion(
        self, context, continuation=None, stop=None, retries=3, delay=5, **kwargs
    ):
        generation_kwargs = self.transform_gen_kwargs(kwargs)
        for _ in range(retries):
            try:
                prompt = context
                request = {
                    "prompt": prompt,
                    "logprobs": self.logprobs,
                    "temperature": self.temperature,
                    **generation_kwargs,
                }
                if continuation:
                    prompt += continuation
                    request.update({"prompt": prompt, "max_tokens": 1, "echo": True})
                if stop is not None:
                    request["stop"] = stop

                response = requests.post(
                    f"{self.base_url}/v1/completions", json=request
                )
                response.raise_for_status()
                return response.json()
            except RequestException as e:
                logger.error(f"RequestException: {e}")
                time.sleep(delay)  # wait before retrying
        else:
            raise Exception(f"Failed to get a valid response after {retries} retries.")

    def transform_gen_kwargs(self, kwargs: Dict) -> Dict:
        """
        Map generation args from task description to GGUF format (OpenAI API format)

        Args:
            kwargs (Dict): kwargs

        Returns:
            Dict: new kwargs
        """
        new_kwargs = kwargs.copy()

        for old_name, new_name in self.kwargs_mapping.items():
            if old_name in new_kwargs.keys():
                new_kwargs[new_name] = new_kwargs.pop(old_name)

        return new_kwargs

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )

    @property
    def chat_template(self) -> str:
        if self.tokenizer.chat_template is not None:
            return self.tokenizer.chat_template
        return self.tokenizer.default_chat_template

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def generate_until(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []

        res = []
        for request in tqdm([req.args for req in requests], disable=disable_tqdm):
            inp = request[0]
            request_args = request[1]
            until = request_args.get("until", ["</s>"])
            response = self.gguf_completion(context=inp, stop=until, **request_args)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "text" in choice:
                    generated_text = choice["text"].strip()
                    res.append(generated_text)
                else:
                    logger.error(
                        f"Invalid response for greedy_until. Response: {response}"
                    )
                    res.append(None)  # Add default value in case of error
            else:
                logger.error(f"Invalid response for greedy_until. Response: {response}")
                res.append(None)  # Add default value in case of error
        return res

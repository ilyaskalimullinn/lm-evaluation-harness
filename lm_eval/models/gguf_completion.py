import logging
import time
from typing import Dict, List

import requests
from requests.exceptions import RequestException
from tqdm import tqdm

from lm_eval.api.registry import register_model
from lm_eval.models.gguf import GGUFLM, get_result
from lm_eval.api.model import LM

from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


@register_model("gguf_completion", "ggml_completion")
class GGUFCompletionLM(LM):
    def __init__(
        self,
        base_url=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.kwargs_mapping = {"max_gen_toks": "max_tokens", "until": "stop"}
        self.base_url = base_url
        self.request_url = f"{base_url}/v1/completions"

        # add default parameters to generation kwargs
        kwargs.update({"temperature": 0.0, "top_logprobs": 10})

        self.base_generation_kwargs = self.transform_gen_kwargs(kwargs)

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

    def gguf_completion(
        self,
        context: str,
        continuation: str = None,
        retries: int = 3,
        delay: int = 5,
        **kwargs,
    ):
        main_request_body = self.create_main_request_body(context)
        generation_kwargs = self.transform_gen_kwargs(kwargs)
        for _ in range(retries):
            try:
                request = {
                    **main_request_body,
                    **self.base_generation_kwargs,
                    **generation_kwargs,
                }
                if continuation:
                    if "prompt" not in request:
                        raise Exception(
                            "Can't use continuation if request has no 'prompt' field"
                        )
                    prompt = request["prompt"]
                    prompt += continuation
                    request.update({"prompt": prompt, "max_tokens": 1, "echo": True})

                response = requests.post(self.request_url, json=request)
                response.raise_for_status()
                return response.json()
            except RequestException as e:
                logger.error(f"RequestException: {e}")
                time.sleep(delay)  # wait before retrying
        else:
            raise Exception(f"Failed to get a valid response after {retries} retries.")

    def create_main_request_body(self, context: str) -> Dict:
        return {"prompt": context}

    def generate_until(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []

        res = []
        for context, request_args in tqdm(
            [req.args for req in requests], disable=disable_tqdm
        ):
            response = self.gguf_completion(context, **request_args)
            resp_result = self.get_result_from_response(response)
            res.append(resp_result)
        return res

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []
        res = []
        for context, continuation in tqdm(
            [req.args for req in requests], disable=disable_tqdm
        ):
            response = self.gguf_completion(
                context=context, continuation=continuation, logprobs=True
            )
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                logprobs = choice.get("logprobs")
                if (
                    logprobs
                    and "token_logprobs" in logprobs
                    and logprobs["token_logprobs"]
                ):
                    logprob, is_greedy = get_result(logprobs, len(context))
                    res.append((logprob, is_greedy))
                else:
                    logger.warning(
                        "Invalid logprobs data. Expected 'logprobs' to contain 'token_logprobs' list."
                    )
            else:
                logger.error(
                    f"Invalid response for loglikelihood. Response: {response}"
                )
                assert False
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for GGUF models"
        )

    def get_result_from_response(self, response):
        if response and "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "text" in choice:
                generated_text = choice["text"].strip()
                return generated_text
            else:
                logger.error(f"Invalid response for greedy_until. Response: {response}")
                return None  # Add default value in case of error
        elif response and "content" in response:
            generated_text = response["content"].strip()
            return generated_text
        else:
            logger.error(f"Invalid response for greedy_until. Response: {response}")
            return None  # Add default value in case of error


@register_model("gguf_completion_chat", "ggml_completion_chat")
class GGUFCompletionChatLM(GGUFCompletionLM):
    def __init__(self, base_url=None, **kwargs) -> None:
        super().__init__(base_url, **kwargs)
        self.request_url = f"{self.base_url}/v1/chat/completions"

    def create_main_request_body(self, context: str) -> Dict:
        return {"messages": [{"role": "user", "content": context}]}

    def get_result_from_response(self, response):
        if response and "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "message" in choice:
                generated_text = choice["message"]["content"].strip()
                return generated_text
            else:
                logger.error(f"Invalid response for greedy_until. Response: {response}")
                return None  # Add default value in case of error
        else:
            logger.error(f"Invalid response for greedy_until. Response: {response}")
            return None  # Add default value in case of error

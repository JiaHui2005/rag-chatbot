import requests
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

class RemoteLLM(LLM):
    api_url: str
    max_new_tokens: int = 512
    temperature: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "remote_colab_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Thực hiện gọi API đến Server Colab."""
        payload = {
            "prompt": prompt,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "mode": kwargs.get("mode", "ft") # Mặc định là fine-tuned
        }
        
        try:
            response = requests.post(f"{self.api_url}/generate", json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "Lỗi: Không nhận được phản hồi từ API.")
        except Exception as e:
            return f"Lỗi khi kết nối đến Colab API: {str(e)}. Vui lòng kiểm tra Server Colab và URL trong file .env"

    @property
    def _identifying_params(self) -> dict:
        return {"api_url": self.api_url}

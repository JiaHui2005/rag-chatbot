import requests
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

class RemoteLLM(LLM):
    api_url: str
    max_new_tokens: int = 512
    temperature: float = 0.3
    mode: str = "ft"

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
        # Đảm bảo URL không có dấu gạch chéo dư thừa
        base_url = self.api_url.rstrip("/")
        
        payload = {
            "prompt": prompt,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "mode": self.mode
        }
        
        try:
            headers = {
                "ngrok-skip-browser-warning": "true",
                "Content-Type": "application/json"
            }
            response = requests.post(f"{base_url}/generate", json=payload, headers=headers, timeout=180)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "Lỗi: Không nhận được phản hồi từ API.")
        except Exception as e:
            return f"Lỗi khi kết nối đến Colab API: {str(e)}. Vui lòng kiểm tra Server Colab và URL trong file .env"

    @property
    def _identifying_params(self) -> dict:
        return {"api_url": self.api_url}

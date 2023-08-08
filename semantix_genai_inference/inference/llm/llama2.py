from typing import Optional
import aiohttp
from .. import SemantixGenClient


class Llama2InferenceClient(SemantixGenClient):
    def __init__(self, inference_server_id: str, api_secret: str, version: Optional[str] = None):
        super().__init__(inference_server_id, api_secret, version=version)
        self._type = "llama2"

    async def predict(self, prompt: str, temperature: Optional[float] = 0.1, top_p: Optional[int] = 40, 
                      top_k: Optional[int] = 80, max_new_tokens: Optional[int] = 1024, 
                      num_beams: Optional[int] = 4):
        body = {
            "prompt": [prompt],
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
        }
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(self.url, json=body) as response:
                return await response.json()
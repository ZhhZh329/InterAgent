import os
from typing import Optional, Dict, List
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class OpenAIClient:
    """异步OpenAI客户端封装"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "qwen-plus",
        default_system_prompt: str = "You are a helpful AI assistant."
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.default_model = default_model
        self.default_system_prompt = default_system_prompt

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or .env file")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 40960,
        **kwargs
    ) -> str:
        """发送聊天请求"""
        full_messages = []
        if system_prompt or self.default_system_prompt:
            full_messages.append({
                "role": "system",
                "content": system_prompt or self.default_system_prompt
            })
        full_messages.extend(messages)

        response = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content


# 全局单例实例
_client = None

def get_client(model_name: Optional[str] = None) -> OpenAIClient:
    """获取全局OpenAI客户端实例"""
    global _client
    if _client is None:
        default_model = model_name or os.getenv('INTER_MODEL', 'qwen-plus')
        _client = OpenAIClient(default_model=default_model)
    return _client


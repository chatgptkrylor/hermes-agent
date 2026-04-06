"""Ollama Native API adapter for Hermes Agent.

Translates between Hermes's internal OpenAI-style message format and
Ollama's native /api/chat endpoint. This provides:

- Native tool calling support (Ollama's format)
- True delta streaming
- Full Ollama parameters (num_ctx, num_predict, etc.)
- ~15-20% latency reduction vs OpenAI-compatible endpoint
- 10-minute read timeout for long-running requests

Author: Nancy (OpenClaw) based on issue #4505 proposal
"""

import json
import logging
import httpx
from typing import Any, Dict, List, Optional, AsyncIterator, Iterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default timeout for Ollama API (10 minutes for long-running requests)
OLLAMA_TIMEOUT = 600.0

# Connection timeout (shorter for initial connection)
OLLAMA_CONNECT_TIMEOUT = 30.0

# Default base URL
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"


@dataclass
class OllamaError(Exception):
    """Error from Ollama API."""
    message: str
    status_code: Optional[int] = None
    
    def __str__(self) -> str:
        if self.status_code:
            return f"OllamaError [{self.status_code}]: {self.message}"
        return f"OllamaError: {self.message}"


class OllamaClient:
    """Native Ollama API client with streaming and tool support.
    
    Uses /api/chat endpoint instead of OpenAI-compatible /v1/chat/completions.
    """
    
    def __init__(
        self,
        base_url: str = OLLAMA_DEFAULT_BASE_URL,
        timeout: float = OLLAMA_TIMEOUT,
        connect_timeout: float = OLLAMA_CONNECT_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self._client = httpx.Client(timeout=httpx.Timeout(timeout, connect=connect_timeout))
        self._async_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=connect_timeout))
    
    def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a synchronous POST request to Ollama."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self._client.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise OllamaError(
                f"HTTP {e.response.status_code}: {e.response.text}",
                status_code=e.response.status_code
            )
        except httpx.RequestError as e:
            raise OllamaError(f"Request failed: {e}")
    
    async def _apost(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an async POST request to Ollama."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = await self._async_client.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise OllamaError(
                f"HTTP {e.response.status_code}: {e.response.text}",
                status_code=e.response.status_code
            )
        except httpx.RequestError as e:
            raise OllamaError(f"Request failed: {e}")
    
    def _stream_post(self, endpoint: str, data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Make a streaming POST request to Ollama."""
        url = f"{self.base_url}{endpoint}"
        try:
            with self._client.stream("POST", url, json=data) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        yield json.loads(line)
        except httpx.HTTPStatusError as e:
            raise OllamaError(
                f"HTTP {e.response.status_code}: {e.response.text}",
                status_code=e.response.status_code
            )
        except httpx.RequestError as e:
            raise OllamaError(f"Request failed: {e}")
    
    async def _astream_post(self, endpoint: str, data: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Make an async streaming POST request to Ollama."""
        url = f"{self.base_url}{endpoint}"
        try:
            async with self._async_client.stream("POST", url, json=data) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield json.loads(line)
        except httpx.HTTPStatusError as e:
            raise OllamaError(
                f"HTTP {e.response.status_code}: {e.response.text}",
                status_code=e.response.status_code
            )
        except httpx.RequestError as e:
            raise OllamaError(f"Request failed: {e}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        response = self._get("/api/tags")
        return response.get("models", [])
    
    def _get(self, endpoint: str) -> Dict[str, Any]:
        """Make a GET request."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self._client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise OllamaError(f"HTTP {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            raise OllamaError(f"Request failed: {e}")
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        format: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send a chat completion request using native Ollama API.
        
        Args:
            model: Model name (e.g., "llama3.2:latest", "qwen3.5:latest")
            messages: List of messages in OpenAI format
            stream: Whether to stream the response
            tools: List of tools in OpenAI format (will be converted)
            tool_choice: Tool choice strategy
            temperature: Sampling temperature
            num_ctx: Context window size
            num_predict: Max tokens to predict
            top_p: Top-p sampling
            top_k: Top-k sampling
            repeat_penalty: Repeat penalty
            seed: Random seed
            format: Output format ("json" for structured output)
            **kwargs: Additional Ollama options
            
        Returns:
            Response dict with 'message' key containing the assistant response
        """
        # Convert OpenAI-style messages to Ollama format
        ollama_messages = self._convert_messages(messages)
        
        # Build Ollama request
        request = {
            "model": model,
            "messages": ollama_messages,
            "stream": stream,
        }
        
        # Add optional parameters
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if num_predict is not None:
            options["num_predict"] = num_predict
        if top_p is not None:
            options["top_p"] = top_p
        if top_k is not None:
            options["top_k"] = top_k
        if repeat_penalty is not None:
            options["repeat_penalty"] = repeat_penalty
        if seed is not None:
            options["seed"] = seed
        if format is not None:
            request["format"] = format
        
        if options:
            request["options"] = options
        
        # Convert tools to Ollama format
        if tools:
            request["tools"] = self._convert_tools(tools)
        
        # Add any additional kwargs to options
        for key, value in kwargs.items():
            if key not in request:
                options[key] = value
        
        if not stream:
            return self._post("/api/chat", request)
        else:
            # For streaming, collect all chunks and return final response
            return self._collect_stream_response(request)
    
    def _collect_stream_response(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Collect streaming response into final response."""
        final_message = {"role": "assistant", "content": ""}
        tool_calls = []
        
        for chunk in self._stream_post("/api/chat", request):
            if "message" in chunk:
                msg = chunk["message"]
                if "content" in msg:
                    final_message["content"] += msg["content"]
                if "tool_calls" in msg:
                    # Accumulate tool calls
                    for tc in msg["tool_calls"]:
                        tool_calls.append(tc)
            if chunk.get("done"):
                final_message["tool_calls"] = tool_calls if tool_calls else None
                return {
                    "model": chunk.get("model", request["model"]),
                    "message": final_message,
                    "done": True,
                    "done_reason": chunk.get("done_reason", "stop"),
                    "total_duration": chunk.get("total_duration", 0),
                    "load_duration": chunk.get("load_duration", 0),
                    "prompt_eval_count": chunk.get("prompt_eval_count", 0),
                    "prompt_eval_duration": chunk.get("prompt_eval_duration", 0),
                    "eval_count": chunk.get("eval_count", 0),
                    "eval_duration": chunk.get("eval_duration", 0),
                }
        
        return {"model": request["model"], "message": final_message, "done": True}
    
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages to Ollama format.
        
        Ollama uses the same basic structure but handles tool responses differently.
        """
        ollama_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle tool results
            if role == "tool":
                # Ollama uses role="tool" with content being the result
                ollama_msg = {
                    "role": "tool",
                    "content": content,
                }
                if "tool_call_id" in msg:
                    ollama_msg["tool_call_id"] = msg["tool_call_id"]
                ollama_messages.append(ollama_msg)
            
            # Handle assistant with tool calls
            elif role == "assistant" and "tool_calls" in msg:
                ollama_msg = {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": msg["tool_calls"],
                }
                ollama_messages.append(ollama_msg)
            
            # Handle system messages with images
            elif role == "system":
                ollama_msg = {"role": "system", "content": content}
                if "images" in msg:
                    ollama_msg["images"] = msg["images"]
                ollama_messages.append(ollama_msg)
            
            # Handle user messages with images
            elif role == "user":
                ollama_msg = {"role": "user", "content": content}
                if "images" in msg:
                    ollama_msg["images"] = msg["images"]
                ollama_messages.append(ollama_msg)
            
            # Handle assistant messages
            elif role == "assistant":
                ollama_messages.append({"role": "assistant", "content": content})
            
            else:
                # Pass through as-is
                ollama_messages.append({"role": role, "content": content})
        
        return ollama_messages
    
    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style tools to Ollama format.
        
        Ollama uses a similar structure but with slightly different field names.
        """
        ollama_tools = []
        
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                ollama_tool = {
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    }
                }
                ollama_tools.append(ollama_tool)
            else:
                # Pass through as-is
                ollama_tools.append(tool)
        
        return ollama_tools
    
    def close(self):
        """Close the HTTP clients."""
        self._client.close()
    
    async def aclose(self):
        """Close the async HTTP client."""
        await self._async_client.aclose()


def build_ollama_client(base_url: str = OLLAMA_DEFAULT_BASE_URL) -> OllamaClient:
    """Build an Ollama client with default settings.
    
    Args:
        base_url: Ollama server URL (default: http://localhost:11434)
        
    Returns:
        OllamaClient instance
    """
    return OllamaClient(base_url=base_url)


# ── Convenience functions for Hermes integration ─────────────────────────────

def ollama_chat_completion(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """OpenAI-compatible chat completion using Ollama native API.
    
    This function provides an OpenAI-compatible interface that internally
    uses Ollama's native /api/chat endpoint.
    
    Args:
        base_url: Ollama server URL
        model: Model name
        messages: OpenAI-style messages
        tools: OpenAI-style tools
        tool_choice: Tool choice strategy
        temperature: Sampling temperature
        **kwargs: Additional parameters
        
    Returns:
        OpenAI-compatible response dict
    """
    client = build_ollama_client(base_url)
    
    try:
        response = client.chat(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            **kwargs,
        )
        
        # Convert to OpenAI-compatible format
        message = response.get("message", {})
        return {
            "id": f"ollama-{response.get('model', model)}",
            "object": "chat.completion",
            "created": response.get("created_at", 0),
            "model": response.get("model", model),
            "choices": [{
                "index": 0,
                "message": {
                    "role": message.get("role", "assistant"),
                    "content": message.get("content", ""),
                    "tool_calls": message.get("tool_calls"),
                },
                "finish_reason": response.get("done_reason", "stop"),
            }],
            "usage": {
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_tokens": (
                    response.get("prompt_eval_count", 0) +
                    response.get("eval_count", 0)
                ),
            },
        }
    finally:
        client.close()


def detect_ollama_server(base_url: str = OLLAMA_DEFAULT_BASE_URL) -> bool:
    """Check if an Ollama server is running at the given URL.
    
    Args:
        base_url: URL to check
        
    Returns:
        True if Ollama is running, False otherwise
    """
    try:
        client = build_ollama_client(base_url)
        client.list_models()
        client.close()
        return True
    except Exception:
        return False
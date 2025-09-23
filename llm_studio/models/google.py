from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import ModelConfig, ProviderAdapter
from ..schemas.messages import Message, ModelResponse, ToolCall
from ..schemas.tooling import ToolSpec


class GoogleProvider(ProviderAdapter):
    """Adapter for Google Gemini models via Generative Language API.

    This is a request/response shaping scaffold with no network calls.
    """

    name = "google"

    def supports_tools(self) -> bool:
        return True

    def prepare_tools(self, tools: List[ToolSpec]) -> List[Dict[str, Any]]:
        # Gemini tools accept a list where each entry is a tool kind
        tool_list: List[Dict[str, Any]] = []

        # Collect function declarations first, then append as a single tool entry if any
        fn_decls: List[Dict[str, Any]] = []
        for t in tools:
            if t.provider == "google" and t.provider_type in {
                "google_search",
                "google_search_retrieval",
            }:
                if t.provider_type == "google_search":
                    tool_list.append({"google_search": {}})
                elif t.provider_type == "google_search_retrieval":
                    entry: Dict[str, Any] = {"google_search_retrieval": {}}
                    cfg = t.provider_config or {}
                    if cfg:
                        entry["google_search_retrieval"] = cfg
                    tool_list.append(entry)
            elif t.provider == "google" and t.provider_type == "code_execution":
                # Enable code execution tool
                tool_list.append({"code_execution": {}})
            elif t.provider == "google" and t.provider_type == "url_context":
                tool_list.append({"url_context": {}})
            else:
                fn_decls.append(
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    }
                )

        if fn_decls:
            tool_list.append({"function_declarations": fn_decls})

        return tool_list

    @staticmethod
    def _map_messages(messages: List[Message]) -> Dict[str, Any]:
        # Gemini uses contents with parts
        system: Optional[str] = None
        contents: List[Dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                if system is None:
                    system = m.content
                continue
            contents.append({"role": m.role, "parts": [{"text": m.content}]})
        return {"system_instruction": system, "contents": contents}

    @staticmethod
    def _parse_response(payload: Dict[str, Any]) -> ModelResponse:
        candidates = payload.get("candidates", [])
        if not candidates:
            return ModelResponse(content="", tool_calls=[], raw=payload)
        cand = candidates[0]
        content = ""
        tool_calls: List[ToolCall] = []

        # Parse different content parts (text, code execution, function calls)
        parts = (cand.get("content") or {}).get("parts") or []
        text_parts = []
        code_parts = []

        for p in parts:
            # Regular text content
            if "text" in p:
                text_parts.append(p.get("text") or "")

            # Function calls (for custom tools)
            elif "function_call" in p:
                fc = p.get("function_call") or {}
                tool_calls.append(
                    ToolCall(
                        id=str(fc.get("id") or len(tool_calls)),
                        name=fc.get("name") or "",
                        arguments=(fc.get("args") or {}),
                    )
                )

            # Code execution parts (executable_code and code_execution_result)
            elif hasattr(p, "executable_code") and p.executable_code:
                code = getattr(p.executable_code, "code", "")
                if code:
                    code_parts.append(f"```python\n{code}\n```")
            elif "executable_code" in p:
                code = p.get("executable_code", {}).get("code", "")
                if code:
                    code_parts.append(f"```python\n{code}\n```")

            elif hasattr(p, "code_execution_result") and p.code_execution_result:
                output = getattr(p.code_execution_result, "output", "")
                if output:
                    code_parts.append(f"Output:\n{output}")
            elif "code_execution_result" in p:
                output = p.get("code_execution_result", {}).get("output", "")
                if output:
                    code_parts.append(f"Output:\n{output}")

        # Combine text and code parts
        all_parts = text_parts + code_parts
        content = "\n".join(part for part in all_parts if part.strip())

        # Extract grounding metadata if present (for Google Search grounding)
        grounding_metadata = cand.get("groundingMetadata")

        # Extract URL context metadata if present (for URL context tool)
        url_context_metadata = cand.get("url_context_metadata")

        # Combine metadata for convenience (both are Google-specific)
        combined_metadata = {}
        if grounding_metadata:
            combined_metadata["grounding"] = grounding_metadata
        if url_context_metadata:
            combined_metadata["url_context"] = url_context_metadata

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            raw=payload,
            grounding_metadata=combined_metadata if combined_metadata else None,
        )

    def generate(
        self,
        messages: List[Message],
        config: ModelConfig,
        tools: Optional[List[ToolSpec]] = None,
    ) -> ModelResponse:
        shaped = self._map_messages(messages)
        request: Dict[str, Any] = {
            "model": config.model,
            "contents": shaped["contents"],
        }
        if shaped.get("system_instruction"):
            request["system_instruction"] = shaped["system_instruction"]
        if tools:
            request["tools"] = self.prepare_tools(tools)
        if config.temperature is not None:
            request["generation_config"] = request.get("generation_config", {})
            request["generation_config"]["temperature"] = config.temperature
        if config.max_tokens is not None:
            request["generation_config"] = request.get("generation_config", {})
            request["generation_config"]["max_output_tokens"] = config.max_tokens
        if config.response_json_schema:
            # Gemini has constrained JSON features; add hint for now
            request["json_schema_hint"] = True

        # Real call to Google Gemini SDK
        try:
            import google.genai as genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)

            # Build config object
            config_obj = None
            if tools:
                config_obj = types.GenerateContentConfig()
                # Convert prepared tools to Google SDK format
                google_tools = []
                for tool_dict in request["tools"]:
                    if "google_search" in tool_dict:
                        google_tools.append(
                            types.Tool(google_search=types.GoogleSearch())
                        )
                    elif "google_search_retrieval" in tool_dict:
                        config_data = tool_dict["google_search_retrieval"]
                        if config_data:
                            drc = config_data.get("dynamic_retrieval_config", {})
                            retrieval_config = types.GoogleSearchRetrieval()
                            if drc:
                                retrieval_config.dynamic_retrieval_config = (
                                    types.DynamicRetrievalConfig(
                                        mode=getattr(
                                            types.DynamicRetrievalConfigMode,
                                            drc.get("mode", "MODE_DYNAMIC"),
                                        ),
                                        dynamic_threshold=drc.get("dynamic_threshold"),
                                    )
                                )
                            google_tools.append(
                                types.Tool(google_search_retrieval=retrieval_config)
                            )
                        else:
                            google_tools.append(
                                types.Tool(
                                    google_search_retrieval=types.GoogleSearchRetrieval()
                                )
                            )
                    elif "code_execution" in tool_dict:
                        google_tools.append(types.Tool(code_execution={}))
                    elif "url_context" in tool_dict:
                        google_tools.append(types.Tool(url_context={}))
                    elif "function_declarations" in tool_dict:
                        func_decls = []
                        for func in tool_dict["function_declarations"]:
                            func_decls.append(
                                types.FunctionDeclaration(
                                    name=func["name"],
                                    description=func["description"],
                                    parameters=types.Schema.from_dict(
                                        func["parameters"]
                                    ),
                                )
                            )
                        google_tools.append(
                            types.Tool(function_declarations=func_decls)
                        )
                config_obj.tools = google_tools

            # Make the API call
            kwargs = {"model": request["model"], "contents": request["contents"]}
            # Note: system_instruction not supported in current SDK version
            if config_obj:
                kwargs["config"] = config_obj

            response = client.models.generate_content(**kwargs)
            return self._parse_response(response.model_dump())

        except ImportError:
            raise RuntimeError(
                "Google GenAI SDK not installed. Install with: pip install google-genai"
            )
        except Exception as e:
            raise RuntimeError(f"Google Gemini API call failed: {str(e)}")

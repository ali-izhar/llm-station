#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, List, Optional

from ..provider import Provider
from ..types import Message, ModelConfig, ModelResponse, ToolCall, ToolSpec


class GoogleProvider(Provider):
    """Adapter for Google Gemini models via Google Gen AI SDK.

    Supports Gemini 2.0+ models with the new search tool and grounding capabilities.
    Uses the latest google-genai SDK for unified access to Google AI and Vertex AI.
    """

    name = "google"

    # Google server-side tool types
    SERVER_TOOL_TYPES = {
        "google_search",
        "code_execution",
        "url_context",
        "image_generation",
    }

    def supports_tools(self) -> bool:
        return True

    def _process_server_tool(self, tool_type: str) -> Optional[Dict[str, Any]]:
        """Process a server-side tool and return its tool dict representation.

        Args:
            tool_type: The type of server tool (google_search, code_execution, url_context)

        Returns:
            Tool dict or None if tool_type is image_generation (handled separately)
        """
        if tool_type == "google_search":
            return {"google_search": {}}
        elif tool_type == "code_execution":
            return {"code_execution": {}}
        elif tool_type == "url_context":
            return {"url_context": {}}
        elif tool_type == "image_generation":
            # Image generation is built into Gemini 2.5 models - no explicit tool needed
            # Just ensure response_modalities include 'Image'
            return None  # Handled in generation config
        return None

    def prepare_tools(self, tools: List[ToolSpec]) -> List[Any]:
        """Convert normalized ToolSpec to Google SDK Tool objects."""
        tool_list: List[Any] = []

        # Function declarations for local tools
        fn_decls: List[Dict[str, Any]] = []

        # Check if we have server tools (google_search, code_execution, etc.)
        has_server_tools = any(
            t.provider == "google" and t.provider_type in self.SERVER_TOOL_TYPES
            for t in tools
        )

        # Check if we have local tools
        has_local_tools = any(
            t.provider != "google" or t.provider_type not in self.SERVER_TOOL_TYPES
            for t in tools
        )

        # Google doesn't support mixing server tools with function tools
        # If we have both, only include server tools
        if has_server_tools and has_local_tools:
            # Only process server tools (exclude image_generation as it's handled separately)
            for t in tools:
                if t.provider == "google" and t.provider_type in self.SERVER_TOOL_TYPES:
                    tool_dict = self._process_server_tool(t.provider_type)
                    if tool_dict:
                        tool_list.append(tool_dict)
            return tool_list

        # Normal processing when not mixing
        for t in tools:
            if t.provider == "google" and t.provider_type in self.SERVER_TOOL_TYPES:
                tool_dict = self._process_server_tool(t.provider_type)
                if tool_dict:
                    tool_list.append(tool_dict)
            else:
                # Local function tools - remove additionalProperties for Google API
                # Also ensure type is a single string, not an array
                tool_schema = dict(t.input_schema)
                tool_schema.pop(
                    "additionalProperties", None
                )  # Google doesn't support this

                # Fix type fields that might be arrays (Google doesn't support union types)
                if "properties" in tool_schema:
                    for prop_name, prop_schema in tool_schema["properties"].items():
                        if isinstance(prop_schema, dict) and "type" in prop_schema:
                            # If type is an array, convert to "object" (most flexible)
                            if isinstance(prop_schema["type"], list):
                                prop_schema["type"] = "object"

                fn_decls.append(
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": tool_schema,
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
        """Parse Google Gemini API response with enhanced grounding metadata support."""
        candidates = payload.get("candidates", [])
        if not candidates:
            return ModelResponse(content="", tool_calls=[], raw=payload)

        cand = candidates[0]
        content = ""
        tool_calls: List[ToolCall] = []

        # Parse different content parts (text, code execution, function calls, images)
        # Optimized: Single pass through parts to extract content and metadata
        parts = (cand.get("content") or {}).get("parts") or []
        text_parts = []
        code_parts = []
        image_parts = []

        # Enhanced metadata extraction for Gemini 2.0+
        grounding_metadata = {}
        code_execution_data = []
        inline_media_data = []
        image_generation_data = []

        # Single pass through parts to extract both content and metadata
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
            elif "executable_code" in p or hasattr(p, "executable_code"):
                exec_code = (
                    p.get("executable_code")
                    if "executable_code" in p
                    else p.executable_code
                )
                if exec_code:
                    # Extract for content display
                    code = (
                        exec_code.get("code")
                        if isinstance(exec_code, dict)
                        else getattr(exec_code, "code", "")
                    )
                    language = (
                        exec_code.get("language", "python")
                        if isinstance(exec_code, dict)
                        else getattr(exec_code, "language", "python")
                    )
                    if code:
                        code_parts.append(f"```{language}\n{code}\n```")

                    # Extract for metadata
                    code_info = {
                        "code": code,
                        "language": language,
                    }
                    code_execution_data.append(code_info)

            elif "code_execution_result" in p or hasattr(p, "code_execution_result"):
                exec_result = (
                    p.get("code_execution_result")
                    if "code_execution_result" in p
                    else p.code_execution_result
                )
                if exec_result:
                    output = (
                        exec_result.get("output")
                        if isinstance(exec_result, dict)
                        else getattr(exec_result, "output", "")
                    )
                    outcome = (
                        exec_result.get("outcome", "OK")
                        if isinstance(exec_result, dict)
                        else getattr(exec_result, "outcome", "OK")
                    )
                    # Extract for content display
                    if output:
                        code_parts.append(f"**Execution Output:**\n```\n{output}\n```")
                    if outcome and outcome != "OK":
                        code_parts.append(f"**Execution Status:** {outcome}")

                    # Extract for metadata
                    result_info = {
                        "output": output,
                        "outcome": outcome,
                    }
                    # Attach to the last code execution entry
                    if code_execution_data:
                        code_execution_data[-1]["result"] = result_info

            # Handle inline_data (images, graphs generated by code)
            elif "inline_data" in p or hasattr(p, "inline_data"):
                inline_data = (
                    p.get("inline_data") if "inline_data" in p else p.inline_data
                )
                if inline_data:
                    mime_type = (
                        inline_data.get("mime_type")
                        if isinstance(inline_data, dict)
                        else getattr(inline_data, "mime_type", "")
                    )
                    # Extract for content display
                    if mime_type and "image" in mime_type:
                        code_parts.append(f"**Generated Image** ({mime_type})")
                    elif mime_type:
                        code_parts.append(f"**Generated Media** ({mime_type})")

                    # Extract for metadata
                    media_info = {
                        "mime_type": mime_type,
                        "data": (
                            inline_data.get("data")
                            if isinstance(inline_data, dict)
                            else getattr(inline_data, "data", "")
                        ),
                        "size": (
                            len(inline_data.get("data", ""))
                            if isinstance(inline_data, dict)
                            else len(getattr(inline_data, "data", ""))
                        ),
                    }
                    inline_media_data.append(media_info)

            # Handle image parts (native image generation)
            elif "image" in p or (hasattr(p, "as_image") and callable(p.as_image)):
                # Gemini 2.5 native image generation
                image_parts.append("**Generated Image**")
                # Note: Actual image data available via response.parts in SDK
                # Extract for metadata
                image_info = {
                    "type": "native_generation",
                    "available": True,
                    "format": "PIL_Image",  # Available via response.parts[].as_image()
                }
                image_generation_data.append(image_info)

            elif "blob" in p and p.get("blob", {}).get("mime_type", "").startswith(
                "image/"
            ):
                # Alternative image format
                blob = p["blob"]
                mime_type = blob.get("mime_type", "")
                image_parts.append(f"**Generated Image** ({mime_type})")
                # Extract for metadata
                image_info = {
                    "type": "blob_image",
                    "mime_type": mime_type,
                    "size": len(blob.get("data", "")),
                    "available": True,
                }
                image_generation_data.append(image_info)

        # Combine all content parts
        all_parts = text_parts + code_parts + image_parts
        content = "\n".join(part for part in all_parts if part.strip())

        # Add metadata if present
        if code_execution_data:
            grounding_metadata["code_execution"] = code_execution_data
        if inline_media_data:
            grounding_metadata["inline_media"] = inline_media_data
        if image_generation_data:
            grounding_metadata["image_generation"] = image_generation_data

        # Extract search grounding metadata (search results, citations)
        raw_grounding = cand.get("grounding_metadata") or cand.get("groundingMetadata")
        if raw_grounding:
            grounding_metadata["grounding"] = raw_grounding

            # Extract search queries if present
            search_queries = []
            gm = raw_grounding
            if "search_entry_point" in gm:
                search_entry_point = gm["search_entry_point"]
                if "rendered_content" in search_entry_point:
                    grounding_metadata["search_entry_point"] = search_entry_point[
                        "rendered_content"
                    ]

            # Extract grounding chunks (search results with citations)
            if "grounding_chunks" in gm:
                chunks = gm["grounding_chunks"]
                grounding_metadata["grounding_chunks"] = chunks

                # Extract sources/citations for compatibility
                sources = []
                citations = []
                for chunk in chunks:
                    if "web" in chunk:
                        web_info = chunk["web"]
                        if "uri" in web_info:
                            sources.append(web_info["uri"])
                        if "title" in web_info:
                            citations.append(
                                {
                                    "url": web_info.get("uri", ""),
                                    "title": web_info.get("title", ""),
                                    "snippet": web_info.get("snippet", ""),
                                }
                            )

                if sources:
                    grounding_metadata["sources"] = sources
                if citations:
                    grounding_metadata["citations"] = citations

        # Extract URL context metadata if present
        if cand.get("url_context_metadata"):
            url_context_data = cand["url_context_metadata"]
            grounding_metadata["url_context"] = url_context_data

            # Extract URL processing status for easier access
            if isinstance(url_context_data, list):
                processed_urls = []
                for url_info in url_context_data:
                    if isinstance(url_info, dict):
                        processed_urls.append(
                            {
                                "url": url_info.get("url", ""),
                                "status": url_info.get("status", ""),
                                "content_type": url_info.get("content_type", ""),
                                "size": url_info.get("size", 0),
                            }
                        )
                if processed_urls:
                    grounding_metadata["processed_urls"] = processed_urls

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            raw=payload,
            grounding_metadata=grounding_metadata if grounding_metadata else None,
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
            from google import genai
            from google.genai.types import GenerateContentConfig, Tool

            client = genai.Client(api_key=self.api_key)

            # Build config object with system instruction and tools
            config_obj = GenerateContentConfig()

            # Add system instruction if present
            if shaped.get("system_instruction"):
                config_obj.system_instruction = shaped["system_instruction"]

            # Add generation config parameters
            if config.temperature is not None:
                config_obj.temperature = config.temperature
            if config.max_tokens is not None:
                config_obj.max_output_tokens = config.max_tokens

            # Check for image generation tools to set response modalities
            has_image_generation = any(
                t.provider == "google" and t.provider_type == "image_generation"
                for t in (tools or [])
            )

            # Set response modalities for image generation
            if has_image_generation:
                config_obj.response_modalities = ["Text", "Image"]

            # Convert prepared tools to Google SDK format
            if tools:
                google_tools = []
                for tool_dict in request["tools"]:
                    if "google_search" in tool_dict:
                        # Gemini 2.0+ search tool - simplified, no configuration needed
                        google_tools.append(Tool(google_search={}))
                    elif "code_execution" in tool_dict:
                        google_tools.append(Tool(code_execution={}))
                    elif "url_context" in tool_dict:
                        google_tools.append(Tool(url_context={}))
                    elif "function_declarations" in tool_dict:
                        # Local function tools
                        func_decls = []
                        for func in tool_dict["function_declarations"]:
                            func_decls.append(
                                {
                                    "name": func["name"],
                                    "description": func["description"],
                                    "parameters": func["parameters"],
                                }
                            )
                        google_tools.append(Tool(function_declarations=func_decls))

                if google_tools:  # Only set tools if we have actual tools to register
                    config_obj.tools = google_tools

            # Make the API call with the new SDK
            response = client.models.generate_content(
                model=request["model"], contents=request["contents"], config=config_obj
            )

            # Convert response to dict format for parsing
            response_dict = (
                response.model_dump() if hasattr(response, "model_dump") else response
            )
            return self._parse_response(response_dict)

        except ImportError:
            raise RuntimeError(
                "Google GenAI SDK not installed. Install with: pip install -U google-genai"
            )
        except (ImportError, AttributeError, KeyError, ValueError) as e:
            raise RuntimeError(f"Google Gemini API call failed: {str(e)}")

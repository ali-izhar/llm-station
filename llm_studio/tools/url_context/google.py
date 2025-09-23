from __future__ import annotations

from ...schemas.tooling import ToolSpec


class GoogleUrlContext:
    """Factory for Google Gemini URL context tool.

    Produces a provider-native ToolSpec that instructs the Google adapter to
    include the `url_context` tool in the request. Provide URLs directly in the
    prompt; Gemini retrieves and uses their content to ground the response.

    Features:
    - Two-step retrieval: internal index cache + live fetch fallback
    - Support for text, images, and PDFs
    - Safety content moderation
    - Up to 20 URLs per request
    - Max 34MB content per URL
    - URL content counted as input tokens

    Usage:
    Include URLs in your prompt and the model will automatically retrieve
    and analyze their content to enhance the response.

    Supported content types:
    - Text: HTML, JSON, plain text, XML, CSS, JavaScript, CSV, RTF
    - Images: PNG, JPEG, BMP, WebP
    - Documents: PDF

    Limitations:
    - No paywalled content
    - No YouTube videos (use video understanding instead)
    - No Google Workspace files
    - No audio/video files
    """

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="url_context",
            description="Google Gemini built-in URL context retrieval tool",
            input_schema={},
            requires_network=True,
            provider="google",
            provider_type="url_context",
            provider_config=None,
        )

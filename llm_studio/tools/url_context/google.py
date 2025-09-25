#!/usr/bin/env python3
from __future__ import annotations

from ...schemas.tooling import ToolSpec


class GoogleUrlContext:
    """Factory for Google Gemini URL context tool for direct content grounding.

    The URL Context tool empowers Gemini models to directly access and process content
    from specific web page URLs provided within API requests. This enables dynamic
    interaction with live web information without manual pre-processing.

    Key Features:
    - **Direct URL Processing**: Access and analyze content from specific web pages
    - **Live Content Retrieval**: Process current website content dynamically
    - **Multi-format Support**: Handle websites, PDFs, and images from URLs
    - **Contextual Grounding**: Anchor model responses to specific URL content
    - **Batch Processing**: Support up to 20 URLs per request
    - **Content Integration**: Retrieved content automatically integrated into context

    How It Works:
    1. Include URLs directly in your prompt text
    2. Enable the URL context tool in your request config
    3. Gemini automatically retrieves and processes the URL content
    4. Model bases its response on the retrieved content

    Supported Content Types:
    - **Websites**: HTML pages, documentation sites, blogs, articles
    - **Documents**: PDF files (reports, papers, manuals)
    - **Images**: PNG, JPEG, BMP, WebP (diagrams, charts, photos)
    - **Data Formats**: JSON, CSV, XML, plain text
    - **Code**: JavaScript, CSS, source code files

    Usage Examples:
        # Website analysis
        agent = Agent(provider="google", model="gemini-2.5-flash", api_key=api_key)
        response = agent.generate(
            "Based on https://example.com/docs, explain the key features",
            tools=["google_url_context"]
        )

        # PDF analysis
        response = agent.generate(
            "Summarize this report: https://example.com/report.pdf",
            tools=["google_url_context"]
        )

        # Image analysis
        response = agent.generate(
            "Describe the components in this diagram: https://example.com/diagram.png",
            tools=["google_url_context"]
        )

        # Combine with search grounding
        response = agent.generate(
            "Analyze this document and search for expert opinions: https://example.com/doc.pdf",
            tools=["google_url_context", "google_search"]
        )

    Technical Details:
    - **URL Limit**: Up to 20 URLs per request
    - **Content Size**: Max 34MB content per URL
    - **Retrieval Method**: Two-step process (cache + live fetch)
    - **Token Counting**: Retrieved content counts toward input tokens
    - **Safety**: Content moderation applied to retrieved content

    Content Limitations:
    - No paywalled or authentication-required content
    - No YouTube videos (use FileData with video URLs instead)
    - No Google Workspace files (Docs, Sheets, etc.)
    - No audio/video files (except as FileData)
    - No content requiring JavaScript execution

    Metadata Access:
    The tool provides retrieval metadata through response.grounding_metadata:
    - url_context_metadata: Status and details of URL retrievals
    - Retrieved URLs and their processing status
    - Content type and size information

    Best Practices:
    - Include URLs naturally in your prompt text
    - Be specific about what you want to analyze from the URLs
    - Combine with search grounding for comprehensive analysis
    - Check url_context_metadata for retrieval status
    """

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="url_context",
            description="Google Gemini URL context tool - directly processes content from web pages, PDFs, and images",
            input_schema={},
            requires_network=True,
            requires_filesystem=False,  # Processes remote content, not local files
            provider="google",
            provider_type="url_context",
            provider_config=None,
        )

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from ...schemas.tooling import ToolSpec


class OpenAIImageGeneration:
    """Factory for OpenAI Image Generation tool (Responses API).

    OpenAI's Image Generation tool allows models to generate or edit images using
    text prompts and optional image inputs. It leverages the GPT Image model and
    automatically optimizes text inputs for improved performance.

    Features:
    - Generate images from text prompts
    - Edit existing images with new prompts
    - Multi-turn iterative editing
    - Streaming partial images (1-3 progressive renders)
    - Automatic prompt revision for better results
    - Base64-encoded image outputs
    - File ID and base64 image inputs

    Output Options:
    - Size: Image dimensions (1024x1024, 1024x1536, auto)
    - Quality: Rendering quality (low, medium, high, auto)
    - Format: File output format (PNG, JPEG, WebP)
    - Compression: Compression level (0-100%) for JPEG/WebP
    - Background: Transparent or opaque (auto)

    Multi-turn Editing:
    - Reference previous response IDs for context
    - Reference specific image generation call IDs
    - Iterative refinement across conversation turns

    Streaming:
    - Progressive image generation (1-3 partial images)
    - Faster visual feedback and improved perceived latency
    - Real-time generation progress

    Args:
        size: Image dimensions. Options: "1024x1024", "1024x1536", "1536x1024",
              "1792x1024", "1024x1792", "auto" (model selects best)
        quality: Rendering quality. Options: "low", "medium", "high", "auto"
        format: Output format. Options: "png", "jpeg", "webp"
        compression: Compression level 0-100 for JPEG/WebP formats
        background: Background type. Options: "transparent", "opaque", "auto"
        partial_images: Number of streaming partial images (1-3)

    Supported Models:
        Mainline models that can call image generation:
        - gpt-4o, gpt-4o-mini
        - gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
        - o3

        Image generation is always performed by gpt-image-1 model.

    Examples:
        Basic generation:
            tool = OpenAIImageGeneration()

        High quality with specific size:
            tool = OpenAIImageGeneration(
                size="1024x1536",
                quality="high",
                format="png"
            )

        Streaming with compression:
            tool = OpenAIImageGeneration(
                quality="medium",
                format="jpeg",
                compression=85,
                partial_images=2
            )

    Usage:
        agent = Agent(provider="openai", model="gpt-5", api="responses")
        response = agent.generate(
            "Generate an image of a gray tabby cat hugging an otter with an orange scarf",
            tools=["openai_image_generation"]
        )

        # Access generated images
        if response.grounding_metadata:
            image_calls = response.grounding_metadata.get("image_generation", [])
            for call in image_calls:
                image_base64 = call["result"]
                revised_prompt = call["revised_prompt"]

    Prompting Tips:
        - Use action words like "draw", "edit", "generate"
        - For editing: "edit the first image by adding..." instead of "combine"
        - Be specific about style, composition, and details
        - The model automatically optimizes prompts for better results
    """

    def __init__(
        self,
        *,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        format: Optional[str] = None,
        compression: Optional[int] = None,
        background: Optional[str] = None,
        partial_images: Optional[int] = None,
    ) -> None:
        """Initialize OpenAI Image Generation tool.

        Args:
            size: Image dimensions or "auto"
            quality: Rendering quality or "auto"
            format: Output format (png, jpeg, webp)
            compression: Compression level 0-100 for JPEG/WebP
            background: Background type or "auto"
            partial_images: Number of streaming partial images (1-3)
        """
        self.size = size
        self.quality = quality
        self.format = format
        self.compression = compression
        self.background = background
        self.partial_images = partial_images

        # Validate inputs
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate image generation configuration parameters."""
        # Validate size
        if self.size is not None:
            valid_sizes = {
                "1024x1024",
                "1024x1536",
                "1536x1024",
                "1792x1024",
                "1024x1792",
                "auto",
            }
            if self.size not in valid_sizes:
                raise ValueError(f"size must be one of {valid_sizes}, got: {self.size}")

        # Validate quality
        if self.quality is not None:
            valid_qualities = {"low", "medium", "high", "auto"}
            if self.quality not in valid_qualities:
                raise ValueError(
                    f"quality must be one of {valid_qualities}, got: {self.quality}"
                )

        # Validate format
        if self.format is not None:
            valid_formats = {"png", "jpeg", "webp"}
            if self.format not in valid_formats:
                raise ValueError(
                    f"format must be one of {valid_formats}, got: {self.format}"
                )

        # Validate compression
        if self.compression is not None:
            if not isinstance(self.compression, int) or not (
                0 <= self.compression <= 100
            ):
                raise ValueError(
                    f"compression must be an integer between 0-100, got: {self.compression}"
                )
            # Compression only valid for JPEG and WebP
            if self.format and self.format not in {"jpeg", "webp"}:
                raise ValueError(
                    f"compression only valid for jpeg/webp formats, got format: {self.format}"
                )

        # Validate background
        if self.background is not None:
            valid_backgrounds = {"transparent", "opaque", "auto"}
            if self.background not in valid_backgrounds:
                raise ValueError(
                    f"background must be one of {valid_backgrounds}, got: {self.background}"
                )

        # Validate partial_images
        if self.partial_images is not None:
            if not isinstance(self.partial_images, int) or not (
                1 <= self.partial_images <= 3
            ):
                raise ValueError(
                    f"partial_images must be an integer between 1-3, got: {self.partial_images}"
                )

    def spec(self) -> ToolSpec:
        """Generate ToolSpec for OpenAI Image Generation tool."""
        provider_config: Dict[str, Any] = {}

        # Add all configuration options
        if self.size is not None:
            provider_config["size"] = self.size
        if self.quality is not None:
            provider_config["quality"] = self.quality
        if self.format is not None:
            provider_config["format"] = self.format
        if self.compression is not None:
            provider_config["compression"] = self.compression
        if self.background is not None:
            provider_config["background"] = self.background
        if self.partial_images is not None:
            provider_config["partial_images"] = self.partial_images

        return ToolSpec(
            name="image_generation",
            description="OpenAI Image Generation tool for creating and editing images (Responses API)",
            input_schema={},  # Not used for provider-native tools
            requires_network=False,  # Server-side generation
            requires_filesystem=False,  # Base64 output, no local files
            provider="openai",
            provider_type="image_generation",
            provider_config=provider_config or None,
        )


class OpenAIImageGenerationAdvanced:
    """Advanced factory for OpenAI Image Generation with multi-turn support.

    Provides explicit support for multi-turn image editing workflows
    with previous response ID and image ID referencing.

    This class demonstrates the advanced patterns but users typically
    use the simpler OpenAIImageGeneration class.
    """

    def __init__(
        self,
        *,
        previous_response_id: Optional[str] = None,
        image_generation_call_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize advanced image generation tool.

        Args:
            previous_response_id: Reference to previous response for context
            image_generation_call_id: Reference to specific image call for editing
            **kwargs: Standard image generation parameters
        """
        self.previous_response_id = previous_response_id
        self.image_generation_call_id = image_generation_call_id
        self.base_tool = OpenAIImageGeneration(**kwargs)

    def spec(self) -> ToolSpec:
        """Generate ToolSpec with multi-turn references."""
        base_spec = self.base_tool.spec()

        # Add multi-turn configuration
        if self.previous_response_id or self.image_generation_call_id:
            config = base_spec.provider_config or {}

            if self.previous_response_id:
                config["previous_response_id"] = self.previous_response_id
            if self.image_generation_call_id:
                config["image_generation_call_id"] = self.image_generation_call_id

            base_spec.provider_config = config

        return base_spec


# Convenience factory functions for common use cases
def create_basic_image_generation() -> ToolSpec:
    """Create basic image generation tool with default settings.

    Returns:
        ToolSpec for basic image generation
    """
    return OpenAIImageGeneration().spec()


def create_high_quality_image_generation(
    size: str = "1024x1536", format: str = "png"
) -> ToolSpec:
    """Create high-quality image generation tool.

    Args:
        size: Image dimensions (default: 1024x1536)
        format: Output format (default: png)

    Returns:
        ToolSpec for high-quality image generation
    """
    return OpenAIImageGeneration(size=size, quality="high", format=format).spec()


def create_streaming_image_generation(
    partial_images: int = 2, quality: str = "medium"
) -> ToolSpec:
    """Create streaming image generation tool with progressive rendering.

    Args:
        partial_images: Number of partial images to stream (1-3)
        quality: Rendering quality

    Returns:
        ToolSpec for streaming image generation
    """
    return OpenAIImageGeneration(partial_images=partial_images, quality=quality).spec()


def create_compressed_image_generation(
    format: str = "jpeg", compression: int = 85, quality: str = "medium"
) -> ToolSpec:
    """Create compressed image generation tool for smaller file sizes.

    Args:
        format: Output format (jpeg or webp)
        compression: Compression level 0-100
        quality: Rendering quality

    Returns:
        ToolSpec for compressed image generation
    """
    return OpenAIImageGeneration(
        format=format, compression=compression, quality=quality
    ).spec()


# Image format and size constants for reference
SUPPORTED_FORMATS = {
    "png": "Portable Network Graphics (lossless, supports transparency)",
    "jpeg": "Joint Photographic Experts Group (lossy, smaller files)",
    "webp": "WebP format (modern, efficient compression)",
}

SUPPORTED_SIZES = {
    "1024x1024": "Square format (1:1 aspect ratio)",
    "1024x1536": "Portrait format (2:3 aspect ratio)",
    "1536x1024": "Landscape format (3:2 aspect ratio)",
    "1792x1024": "Wide landscape format (7:4 aspect ratio)",
    "1024x1792": "Tall portrait format (4:7 aspect ratio)",
    "auto": "Model automatically selects best size for prompt",
}

QUALITY_LEVELS = {
    "low": "Fast generation, lower detail",
    "medium": "Balanced speed and quality",
    "high": "Slower generation, maximum detail",
    "auto": "Model automatically selects best quality for prompt",
}

BACKGROUND_OPTIONS = {
    "transparent": "Transparent background (PNG format recommended)",
    "opaque": "Solid background color",
    "auto": "Model automatically selects best background type",
}

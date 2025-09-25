#!/usr/bin/env python3
from __future__ import annotations

from ...schemas.tooling import ToolSpec


class GoogleImageGeneration:
    """Factory for Google Gemini 2.5 native image generation capability.

    Gemini 2.5 models include native image generation capabilities that allow multimodal
    output combining both text and images in a single response. This is different from
    standalone image generation models as it's integrated into the conversational flow.

    Key Features:
    - **Character Consistency**: Preserve subject appearance across multiple generated images
    - **Intelligent Editing**: Precise prompt-based edits like inpainting and outpainting
    - **Image Composition**: Combine elements from multiple input images
    - **Multimodal Reasoning**: Understand visual context and follow complex instructions
    - **Iterative Generation**: Perfect for chat-based image refinement workflows
    - **Multiple Images**: Generate multiple images in a single response (stories, tutorials)

    Capabilities:
    - **Image Creation**: Generate photorealistic or artistic images from text prompts
    - **Image Editing**: Modify existing images with natural language instructions
    - **Story Telling**: Create multi-image narratives and step-by-step tutorials
    - **Character Development**: Maintain consistent characters across scenes
    - **Scene Composition**: Combine multiple characters/objects from different images
    - **Style Transfer**: Apply artistic styles and transformations

    Model Requirements:
    - Uses gemini-2.5-flash-image-preview (latest model with image generation)
    - Requires billing enabled (pay-as-you-go feature)
    - Supports response_modalities=['Text', 'Image'] for mixed output

    Usage Examples:
        # Basic image generation
        agent = Agent(provider="google", model="gemini-2.5-flash-image-preview", api_key=api_key)
        response = agent.generate(
            "Create a photorealistic image of a siamese cat with heterochromia",
            tools=["google_image_generation"]
        )

        # Image editing with reference
        response = agent.generate(
            "Edit this cat to be in a tropical forest eating a banana",
            tools=["google_image_generation"]
        )

        # Multi-image story generation
        response = agent.generate(
            "Create a 5-part visual story about a robot's adventure in space",
            tools=["google_image_generation"]
        )

        # Character consistency across scenes
        chat = client.chats.create(model="gemini-2.5-flash-image-preview")
        response1 = chat.send_message("Create a toy fox figurine")
        response2 = chat.send_message("Now show that same figurine on a beach")

    Response Structure:
    The response contains multiple parts that can be:
    - text: Descriptive text about the generation process
    - image: Generated image data (PIL Image format)
    - Multiple images in a single response for stories/tutorials

    Best Practices:
    - Use chat mode for iterative image refinement
    - Be specific about character details to maintain consistency
    - Combine with multimodal inputs for editing scenarios
    - Request multiple images for storytelling or tutorials
    - Use descriptive prompts for better generation quality

    Technical Details:
    - Integrated into Gemini 2.5 model architecture
    - No separate tool configuration needed
    - Images accessible via response.parts with as_image() method
    - Supports PIL Image operations for saving/processing
    - Pay-per-generation pricing model

    Limitations:
    - Requires gemini-2.5-flash-image-preview model
    - Billing must be enabled for image generation
    - Generation speed depends on image complexity
    - Content policies apply to generated images

    Advanced Features:
    - Mix up to 3 input images for composition
    - Maintain character consistency across generations
    - Support for various artistic styles and formats
    - Integration with code execution for image processing workflows
    """

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="image_generation",
            description="Google Gemini 2.5 native image generation - creates and edits images with multimodal reasoning",
            input_schema={},
            requires_network=False,  # Built into model
            requires_filesystem=False,  # Image data in response
            provider="google",
            provider_type="image_generation",
            provider_config=None,
        )


class GoogleImageGenerationChat:
    """Factory for chat-based iterative image generation with Gemini 2.5.

    Optimized for multi-turn conversations where images are refined and edited
    across multiple interactions. This is the recommended approach for most
    image generation workflows.

    Features:
    - Maintains image context across conversation turns
    - Enables iterative refinement and editing
    - Character and object consistency across generations
    - Natural language editing commands
    - Progressive story building with images

    Usage:
        # Start a chat session for iterative image generation
        agent = Agent(provider="google", model="gemini-2.5-flash-image-preview", api_key=api_key)

        # Initial generation
        response1 = agent.generate("Create a space explorer character")

        # Iterative refinement
        response2 = agent.generate("Add a helmet with a blue planet design")
        response3 = agent.generate("Move the character to a beach setting")
        response4 = agent.generate("Now show them base-jumping from a spaceship")
    """

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="image_generation_chat",
            description="Google Gemini 2.5 chat-based image generation for iterative refinement",
            input_schema={},
            requires_network=False,
            requires_filesystem=False,
            provider="google",
            provider_type="image_generation",
            provider_config={"chat_optimized": True},
        )


# Convenience factory functions for common use cases
def create_basic_image_generation() -> ToolSpec:
    """Create basic image generation tool with default settings."""
    return GoogleImageGeneration().spec()


def create_chat_image_generation() -> ToolSpec:
    """Create chat-optimized image generation tool."""
    return GoogleImageGenerationChat().spec()


def create_story_image_generation() -> ToolSpec:
    """Create image generation tool optimized for multi-image storytelling."""
    tool_spec = GoogleImageGeneration().spec()
    tool_spec.description += " - optimized for multi-image story generation"
    return tool_spec


# Model and capability constants for reference
SUPPORTED_MODELS = {
    "gemini-2.5-flash-image-preview": "Latest model with native image generation",
    "gemini-2.5-pro-image-preview": "Advanced model with enhanced image capabilities",
}

IMAGE_GENERATION_FEATURES = {
    "character_consistency": "Maintain subject appearance across images",
    "intelligent_editing": "Precise prompt-based image modifications",
    "image_composition": "Combine elements from multiple input images",
    "multimodal_reasoning": "Understand visual context and complex instructions",
    "iterative_generation": "Perfect for chat-based refinement workflows",
    "multi_image_output": "Generate multiple images in single response",
}

BEST_PRACTICES = {
    "use_chat_mode": "Recommended for iterative image refinement",
    "specific_prompts": "Be detailed about character features for consistency",
    "combine_modalities": "Mix text and image inputs for editing",
    "request_multiple": "Ask for multiple images for stories/tutorials",
    "descriptive_language": "Use vivid descriptions for better quality",
}

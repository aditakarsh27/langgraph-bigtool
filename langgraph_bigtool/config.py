from pydantic import BaseModel, Field
from typing import Annotated, Literal

# Literal type for all available usecase filenames
UseCaseName = Literal[
    " ",
    "1-unsubscribe",
    "2-reddit", 
    "2.1-reddit",
    "3-earnings",
    "4-maps",
    "4.1-maps",
    "5-gmailreply",
    "6-contract",
    "7-overnight",
    "8-sheets_chart",
    "9-learning",
    "10-reddit2",
    "11-github"
]

class ContextSchema(BaseModel):
    """The configuration for the agent."""

    base_prompt: str = Field(
        default=" ",
        description="The base prompt to use for the agent's interactions. Leave blank if using a JSON prompt from the dropdown."
    )
    model_provider: Annotated[
        Literal[
            "openai",
            "anthropic",
            "azure_openai",
            "azure_ai",
            "google_vertexai",
            "google_genai",
            "bedrock",
            "bedrock_converse",
            "cohere",
            "fireworks",
            "together",
            "mistralai",
            "huggingface",
            "groq",
            "ollama",
            "google_anthropic_vertex",
            "deepseek",
            "ibm",
            "nvidia",
            "xai",
            "perplexity"
        ],
        {"__template_metadata__": {"kind": "provider"}},
    ] = Field(
        default="anthropic",
        description="The name of the model provider to use for the agent's main interactions. ",
    )
    model: Annotated[
        Literal[
            "claude-4-sonnet-20250514",
            "claude-sonnet-4@20250514",
        ],
        {"__template_metadata__": {"kind": "llm"}},
    ] = Field(
        default="claude-4-sonnet-20250514",
        description="The name of the language model to use for the agent's main interactions. ",
    )
    tool_names: list[str] = Field(
        default=[],
        description="The names of the tools to use for the agent's main interactions. Leave blank if using a JSON prompt from the dropdown."
    )
    json_prompt_name: UseCaseName = Field(
        default=" ",
        description="The name of the JSON prompt to use for the agent's main interactions, instead of providing a base prompt and tool names. "
    )


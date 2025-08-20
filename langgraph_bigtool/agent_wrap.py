


import math
import types
import uuid
from langchain_core.runnables import RunnableConfig

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

from langgraph_bigtool import create_agent
from langgraph_bigtool.config import ContextSchema

from langgraph_bigtool.utils import (
    convert_positional_only_function_to_tool
)

from universal_mcp.agentr import Agentr
from universal_mcp.types import ToolFormat
import types
import yaml
from langchain.chat_models import init_chat_model
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_anthropic import ChatAnthropic


async def agent(config: RunnableConfig):
    cfg = ContextSchema(**config.get("configurable", {}))

    all_tools = []
    for function_name in dir(math):
        function = getattr(math, function_name)
        if not isinstance(
            function, types.BuiltinFunctionType
        ):
            continue
        # This is an idiosyncrasy of the `math` library
        if tool := convert_positional_only_function_to_tool(
            function
        ):
            all_tools.append(tool)

    # Create registry of tools. This is a dict mapping
    # identifiers to tool instances.
    tool_registry = {
        str(uuid.uuid4()): tool
        for tool in all_tools
    }

    # Index tool names and descriptions in the LangGraph
    # Store. Here we use a simple in-memory store.
    embeddings = init_embeddings("openai:text-embedding-3-small")

    store = InMemoryStore(
        index={
            "embed": embeddings,
            "dims": 1536,
            "fields": ["description"],
        }
    )
    for tool_id, tool in tool_registry.items():
        store.put(
            ("tools",),
            tool_id,
            {
                "description": f"{tool.name}: {tool.description}",
            },
        )

    # Initialize agent
    llm = init_chat_model("openai:gpt-4o-mini")
    print(tool_registry)

    builder = create_agent(llm, tool_registry)
    return builder.compile(store=store)

    # if cfg.json_prompt_name and cfg.json_prompt_name.strip():
    #     with open(f"usecases/{cfg.json_prompt_name}.yaml", "r", encoding='utf-8') as f:
    #         content = f.read()
    #         data = yaml.safe_load(content)
    #         if cfg.base_prompt and cfg.base_prompt.strip():
    #             pass
    #         else:
    #             cfg.base_prompt = data["base_prompt"]
    #         cfg.tool_names = data["tools"]  
    # agentr = Agentr()
    # agentr.load_tools(cfg.tool_names)
    # tools = [] # can add custom tools here like get_weather, get_simple_weather, etc.

    # tools_agentr = agentr.list_tools(format=ToolFormat.NATIVE)
    # tools.extend(tools_agentr)

    # if cfg.model_provider == "google_anthropic_vertex":
    #     # For Google Anthropic Vertex, we need to use the specific model initialization due to location
    #     model = ChatAnthropicVertex(model=cfg.model, temperature=0.2, location="asia-east1")
    # elif cfg.model == "claude-4-sonnet-20250514":
    #     model = ChatAnthropic(model=cfg.model, temperature=1, thinking={"type": "enabled", "budget_tokens": 2048}, max_tokens=4096) # pyright: ignore[reportCallIssue]
    # else:
    #     model = init_chat_model(model=cfg.model, model_provider=cfg.model_provider, temperature=0.2)


    # code_act = create_agent(model, cfg.base_prompt, tools, eval)
    # return code_act.compile()



    

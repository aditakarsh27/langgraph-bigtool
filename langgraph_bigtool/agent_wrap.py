


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
from universal_mcp.applications import app_from_slug
from langchain_core.tools import StructuredTool
import inspect
import types
import yaml
from langchain.chat_models import init_chat_model
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver



async def agent(config: RunnableConfig):
    cfg = ContextSchema(**config.get("configurable", {}))

    all_tools = []
    
    # Use Agentr to fetch tools instead of math module
    agentr = Agentr()
    
    # Load tools based on configuration
    if cfg.json_prompt_name and cfg.json_prompt_name.strip():
        with open(f"usecases/{cfg.json_prompt_name}.yaml", "r", encoding='utf-8') as f:
            content = f.read()
            data = yaml.safe_load(content)
            if cfg.base_prompt and cfg.base_prompt.strip():
                pass
            else:
                cfg.base_prompt = data["base_prompt"]
            cfg.tool_names = data["tools"]
    
    # Default tool names if none specified
    if not cfg.tool_names:
        cfg.tool_names = ['google-mail_list_messages', 'google-mail_get_message', 'google-mail_get_messages', 'perplexity_chat', 'serpapi_google_maps_search', 'firecrawl_scrape_url', 'google-drive_get_file', 'google-drive_create_folder', 'google-drive_find_folder_id_by_name', 'google-drive_list_files', 'google-sheet_update_values', 'google-sheet_get_values', 'google-sheet_get_spreadsheet', 'google-sheet_batch_get_values', 'google-sheet_create_spreadsheet', 'google-sheet_clear_values', 'hubspot_search_contacts_post', 'hubspot_batch_read_contacts_post', 'hubspot_get_contacts', 'hubspot_get_contact_by_id', 'hubspot_update_contact_by_id', 'hubspot_batch_update_contacts', 'hubspot_create_contacts_batch', 'hubspot_create_contact',
        'google-sheet_create_spreadsheet', 'google-sheet_get_spreadsheet', 'google-sheet_batch_get_values', 'google-sheet_append_dimensions',
        'google-sheet_insert_dimensions', 'google-sheet_delete_sheet', 'google-sheet_add_sheet', 'google-sheet_delete_dimensions',
        'google-sheet_add_basic_chart', 'google-sheet_add_table', 'google-sheet_add_pie_chart', 'google-sheet_clear_values',
        'google-sheet_update_values', 'google-sheet_clear_basic_filter', 'google-sheet_batch_update', 'google-sheet_get_values',
        'google-sheet_list_tables', 'google-sheet_set_basic_filter', 'google-sheet_get_table_schema', 'google-sheet_copy_to_sheet',
        'google-sheet_append_values', 'google-sheet_batch_get_values_by_data_filter', 'google-sheet_batch_clear_values', 'google-sheet_format_cells', 'reddit_get_post_comments_details', 'clickup_comments_get_task_comments', 'clickup_comments_get_list_comments', 'clickup_comments_get_view_comments', 'clickup_tasks_get_list_tasks', 'clickup_tasks_filter_team_tasks',
        'clickup_time_tracking_get_time_entries_within_date_range', 'clickup_time_tracking_get_time_entry_history', 'clickup_authorization_get_workspace_list', 'clickup_spaces_get_space_details', 'clickup_lists_get_list_details'
        ]
    
    # Load tools from Agentr
    try:
        agentr.load_tools(cfg.tool_names)
        tools_agentr = agentr.list_tools(format=ToolFormat.NATIVE)
        print(f"Loaded {len(tools_agentr)} tools from Agentr")
        
        # Convert tools to StructuredTool format
        tools_agentr = [
            t if isinstance(t, StructuredTool)
            else StructuredTool.from_function(coroutine=t) if inspect.iscoroutinefunction(t)
            else StructuredTool.from_function(func=t)
            for t in tools_agentr
        ]
        all_tools.extend(tools_agentr)
        
        for tool in all_tools:
            print(f"  - {tool.name}: {tool.description}")
    except Exception as e:
        print(f"Error loading Agentr tools: {e}")
        # Fallback to math tools if Agentr fails
        print("Falling back to math tools...")
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

    print(f"Tool registry size: {len(tool_registry)}")

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

    # Initialize agent with appropriate model
    if cfg.model_provider == "google_anthropic_vertex":
        # For Google Anthropic Vertex, we need to use the specific model initialization due to location
        llm = ChatAnthropicVertex(model=cfg.model, temperature=0.2, location="asia-east1")
    elif cfg.model == "claude-4-sonnet-20250514":
        llm = ChatAnthropic(model=cfg.model, temperature=1, thinking={"type": "enabled", "budget_tokens": 2048}, max_tokens=4096) # pyright: ignore[reportCallIssue]
    else:
        llm = init_chat_model(model=cfg.model, model_provider=cfg.model_provider, temperature=0.2)

    print(f"Final tool_registry: {tool_registry}")
    checkpointer=MemorySaver()

    builder = create_agent(llm, tool_registry)
    return builder.compile(store=store, checkpointer=checkpointer)



    

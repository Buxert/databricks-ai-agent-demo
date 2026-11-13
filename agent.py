import os, json, uuid, asyncio, logging, time
import mlflow
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, InjectedStore
from langgraph.store.base import BaseStore
from databricks_langchain import ChatDatabricks, AsyncCheckpointSaver, AsyncDatabricksStore
from typing import Annotated

logger = logging.getLogger(__name__)

# --- Config from environment ---
_endpoint = os.environ.get("SERVING_ENDPOINT_NAME", "databricks-gpt-oss-120b")
_catalog = os.environ.get("CATALOG_NAME", "")
_schema = os.environ.get("SCHEMA_NAME", "spotify")
_lb_project = os.environ.get("LAKEBASE_AUTOSCALING_PROJECT", "")
_lb_branch = os.environ.get("LAKEBASE_AUTOSCALING_BRANCH", "production")
_experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID", "")
_func1 = os.environ.get("FUNCTION_1_NAME", "top_artists")
_func2 = os.environ.get("FUNCTION_2_NAME", "most_played_tracks")
_wh = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
_sql_tables = [t for t in (
    os.environ.get("PLAY_HISTORY_TABLE", "labuser_benoit.spotify.play_history"),
) if t]
_embedding_endpoint = os.environ.get("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
_embedding_dims = int(os.environ.get("EMBEDDING_DIMS", "1024"))

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", """You are DataRocker, a personal Spotify listening analyst. You help the user explore and understand their own Spotify listening behavior: favorite artists and tracks, listening trends over time, and fun patterns in their history.

At the START of every conversation:
1. Call recall_memories to check what you already know about this user
2. If memories exist, greet them by name and reference their music taste

When you learn new facts about the user (name, favorite genres, preferred artists, how they like insights presented), call save_memory to store them.

The listening history is polled hourly from the Spotify API. Each row in play_history is one play event: played_at (UTC timestamp), track_name, primary_artist, artist_names (JSON array of all artists), album_name, duration_ms (milliseconds), and context (album/playlist/track). History collection started in November 2026, so earlier periods have no data.

Always be specific with numbers and provide actionable, fun insights.

You also have direct READ-ONLY SQL access to the Spotify listening tables (play_history). Use list_table_schemas to discover the available tables and their columns, then run_sql to run SELECT queries for any data question the dedicated tools don't cover. You can only read data — never attempt to create, update, or delete anything.""")

# MLflow setup
mlflow.set_tracking_uri("databricks")
mlflow.langchain.autolog()
if _experiment_id:
    try:
        mlflow.set_experiment(experiment_id=_experiment_id)
    except Exception as exc:
        # Tracing is optional - never block app startup on experiment access
        print(f"MLflow tracing disabled (could not access experiment): {exc}")

# --- Memory tools ---
@tool
def save_memory(key: str, value: str, config: RunnableConfig, store: Annotated[BaseStore, InjectedStore]) -> str:
    """Save a fact about this listener (name, favorite genres, preferred artists, etc.)."""
    user_id = config["configurable"].get("user_id", "anonymous")
    store.put(("listeners", user_id), key, {"content": value})
    return f"Saved '{key}' = '{value}'"

@tool
def recall_memories(query: str, config: RunnableConfig, store: Annotated[BaseStore, InjectedStore]) -> str:
    """Search this user's stored facts by semantic similarity."""
    user_id = config["configurable"].get("user_id", "anonymous")
    results = store.search(("listeners", user_id), query=query, limit=5)
    if not results:
        return "No memories found for this user."
    return "Recalled:\n" + "\n".join(f"  [{r.key}]: {r.value['content']}" for r in results)

# --- MCP tool discovery ---
_mcp_tools = None

async def _get_mcp_tools():
    global _mcp_tools
    if _mcp_tools is not None:
        return _mcp_tools
    _mcp_tools = []
    if _catalog and _schema and _func1:
        try:
            try:
                from unitycatalog.ai.langchain.toolkit import UCFunctionToolkit
            except ImportError:
                from unitycatalog.ai.langchain_toolkit import UCFunctionToolkit
            names = [f"{_catalog}.{_schema}.{_func1}"]
            if _func2:
                names.append(f"{_catalog}.{_schema}.{_func2}")
            toolkit = UCFunctionToolkit(function_names=names)
            _mcp_tools = toolkit.tools
            for t in _mcp_tools:
                t.name = t.name.replace(".", "_")
        except Exception as e:
            logger.warning(f"UC tools not loaded: {e}")
    return _mcp_tools

# --- Read-only SQL tools (direct table access) ---
_sql_tools = None
_schema_cache = None

def _state_str(r):
    s = r.status.state
    v = getattr(s, "value", None)
    if v is None:
        v = str(s).split(".")[-1]
    return v

def _exec_sql(stmt, max_rows=50):
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    r = w.statement_execution.execute_statement(
        statement=stmt, warehouse_id=_wh, wait_timeout="50s"
    )
    deadline = time.time() + 60
    while _state_str(r) in ("PENDING", "RUNNING") and time.time() < deadline:
        time.sleep(2)
        r = w.statement_execution.get_statement(r.statement_id)
    if _state_str(r) != "SUCCEEDED":
        return f"Query did not succeed (state: {_state_str(r)})."
    if not r.result or not r.result.data_array:
        return "Query succeeded (no rows returned)."
    cols = [c.name for c in r.manifest.schema.columns]
    rows = []
    for item in list(r.result.data_array)[:max_rows]:
        vals = getattr(item, "values", None)
        if vals is None:
            vals = item
        rows.append(dict(zip(cols, list(vals))))
    out = json.dumps(rows, default=str)
    if len(r.result.data_array) > max_rows:
        out += f"\n(showing first {max_rows} of {len(r.result.data_array)} rows)"
    return out

@tool
def run_sql(query: str) -> str:
    """Run a READ-ONLY SQL query (SELECT or WITH only) against the Spotify listening history in Unity Catalog.
    Call list_table_schemas first to learn table names and columns. Add LIMIT when exploring large tables.
    Returns up to 50 rows as JSON."""
    q = query.strip().rstrip(";")
    if not (q.lower().startswith("select") or q.lower().startswith("with")):
        return "Error: only read-only SELECT/WITH queries are allowed."
    try:
        return _exec_sql(q)
    except Exception as e:
        return f"SQL error: {e}"

@tool
def list_table_schemas() -> str:
    """List the Spotify listening history tables available for SQL queries, with their columns and row counts.
    Call this before writing any SQL."""
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache
    parts = []
    for t in _sql_tables:
        try:
            desc = _exec_sql(f"DESCRIBE TABLE {t}", max_rows=200)
            cnt = _exec_sql(f"SELECT COUNT(*) AS row_count FROM {t}", max_rows=1)
            parts.append(f"TABLE {t}\ncolumns: {desc}\ncount: {cnt}")
        except Exception as e:
            parts.append(f"TABLE {t}: unavailable ({e})")
    _schema_cache = "\n\n".join(parts) if parts else "No tables available."
    return _schema_cache

def _get_sql_tools():
    global _sql_tools
    if _sql_tools is not None:
        return _sql_tools
    _sql_tools = [run_sql, list_table_schemas] if (_wh and _sql_tables) else []
    return _sql_tools

# --- Graph builder ---
def _build_graph(checkpointer, store, model_with_tools, tools):
    def agent_node(state: MessagesState):
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        return {"messages": [model_with_tools.invoke(msgs)]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    wf = StateGraph(MessagesState)
    wf.add_node("agent", agent_node)
    # handle_tool_errors: a failing tool returns the error to the LLM so it can
    # retry with corrected arguments instead of crashing the whole request
    wf.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    wf.add_edge(START, "agent")
    wf.add_conditional_edges("agent", should_continue, ["tools", END])
    wf.add_edge("tools", "agent")
    return wf.compile(checkpointer=checkpointer, store=store)

# --- Async execution ---
async def _run_async(messages, thread_id, user_id, model, tools):
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    lk = {"project": _lb_project, "branch": _lb_branch}
    mem_tools = [t for t in tools if t.name in ("save_memory", "recall_memories")]
    other_tools = [t for t in tools if t.name not in ("save_memory", "recall_memories")]

    async def _invoke(model_with_tools, graph_tools, checkpointer=None, store=None):
        graph = _build_graph(checkpointer, store, model_with_tools, graph_tools)
        result = await graph.ainvoke({"messages": messages}, config)
        return result["messages"][-1].content

    # Preferred path: Lakebase-backed memory (checkpointer + store)
    if _lb_project:
        try:
            async with AsyncDatabricksStore(**lk, embedding_endpoint=_embedding_endpoint, embedding_dims=_embedding_dims) as store:
                await store.setup()
                async with AsyncCheckpointSaver(**lk) as cp:
                    return await _invoke(model.bind_tools(tools), tools, cp, store)
        except Exception as exc:
            print(f"Memory unavailable ({type(exc).__name__}: {str(exc)[:200]}); running without memory.")

    # Fallback: no memory - still fully functional analytics agent
    return await _invoke(model.bind_tools(other_tools), other_tools)

# --- Public interface ---
@mlflow.trace(name="music_agent_run", span_type="AGENT")
async def run_agent(messages, thread_id=None, user_id="anonymous"):
    thread_id = thread_id or str(uuid.uuid4())
    mcp_tools = await _get_mcp_tools()
    sql_tools = _get_sql_tools()
    all_tools = [save_memory, recall_memories] + mcp_tools + sql_tools

    model = ChatDatabricks(endpoint=_endpoint)
    model_with_tools = model.bind_tools(all_tools)

    output = await _run_async(messages, thread_id, user_id, model, all_tools)
    return {"output": output, "thread_id": thread_id}

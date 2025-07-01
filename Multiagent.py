from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from langchain_community.llms import Ollama
from langchain.agents import Tool, initialize_agent
from langgraph.graph import StateGraph, END
from typing import TypedDict
import uvicorn

# --- LangChain & LangGraph setup

llm = Ollama(model="llama3")

# Tool logic
def read_it_docs(input: str) -> str:
    return f"[IT DOCS] Here's how to {input}"

def read_finance_docs(input: str) -> str:
    return f"[Finance DOCS] Here's how to {input}"

def web_search(input: str) -> str:
    return f"[WEB SEARCH] Couldn't find internally. Here's external info on {input}"

read_it_tool = Tool(name="ReadITDocs", func=read_it_docs, description="Query internal IT documentation.")
read_finance_tool = Tool(name="ReadFinanceDocs", func=read_finance_docs, description="Query internal Finance documentation.")
web_search_tool = Tool(name="WebSearch", func=web_search, description="Search the web for external info.")

class AgentState(TypedDict):
    query: str
    classification: str
    answer: str

def supervisor_node(state: AgentState) -> dict:
    query = state["query"].lower()
    if any(k in query for k in ["vpn", "laptop", "email", "wifi", "software"]):
        return {"classification": "IT"}
    elif any(k in query for k in ["payroll", "invoice", "reimbursement", "budget"]):
        return {"classification": "Finance"}
    return {"classification": "IT"}

def it_agent(state: AgentState) -> dict:
    agent = initialize_agent(
        tools=[read_it_tool, web_search_tool],
        llm=llm,
        agent_type="zero-shot-react-description",
        verbose=False
    )
    result = agent.run(state["query"])
    return {"answer": result}

def finance_agent(state: AgentState) -> dict:
    agent = initialize_agent(
        tools=[read_finance_tool, web_search_tool],
        llm=llm,
        agent_type="zero-shot-react-description",
        verbose=False
    )
    result = agent.run(state["query"])
    return {"answer": result}

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("it_agent", it_agent)
graph.add_node("finance_agent", finance_agent)
graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", lambda s: s["classification"], {
    "IT": "it_agent",
    "Finance": "finance_agent"
})
graph.add_edge("it_agent", END)
graph.add_edge("finance_agent", END)
workflow = graph.compile()

# --- FastAPI web app setup

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
def form_post(request: Request, query: str = Form(...)):
    response = workflow.invoke({"query": query})
    answer = response["answer"]
    return templates.TemplateResponse("form.html", {"request": request, "result": answer, "query": query})


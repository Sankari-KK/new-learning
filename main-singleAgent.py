#Build an agent using LangChain with access to two tools:

#WriteFile: Save research output in Markdown format
#WebSearch: Search the web to gather content

#Use this agent to respond to a user research question and write the result to a file
    

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import os
import webbrowser
import pdfkit
import markdown as md
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

# Load environment
load_dotenv()

# FastAPI setup
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load prompt template
with open("agent_prompt.txt", "r", encoding="utf-8") as f:
    prompt_template_str = f.read()

# Define tools
search = DuckDuckGoSearchRun()
web_search_tool = Tool(
    name="WebSearch",
    func=search.run,
    description="Useful for finding current events and information on the internet."
)

def write_to_md(content: str) -> str:
    print("📦 Content received by WriteFile:\n", content)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    md_filename = f"sankari_output_{timestamp}.md"
    pdf_filename = f"sankari_output_{timestamp}.pdf"

    formatted_content = f"""# 🔍 Research Summary

## 🧠 Answer:
{content}

---

_Generated using LangChain agent with web search._
"""

    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(formatted_content)

    html_content = md.markdown(formatted_content)
    pdfkit.from_string(html_content, pdf_filename)

    webbrowser.open(f"file://{os.path.abspath(md_filename)}")

    return f"✅ Markdown and PDF created: {md_filename}, {pdf_filename}"

write_file_tool = Tool(
    name="WriteFile",
    func=write_to_md,
    description="Writes research to Markdown, converts to PDF, and opens it in browser."
)

tools = [web_search_tool, write_file_tool]

# Create agent prompt
prompt = PromptTemplate.from_template(prompt_template_str).partial(
    tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
    tool_names=", ".join([tool.name for tool in tools])
)

# Create agent
llm = Ollama(model="llama3")
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Web form UI
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index1.html", {"request": request, "response": ""})

@app.post("/", response_class=HTMLResponse)
async def handle_query(request: Request, query: str = Form(...)):
    try:
        result = agent_executor.invoke({"input": query})
        print("💬 Full agent result:", result)
        response = result.get("output", "No output returned.")
    except Exception as e:
        response = f"❌ Agent error: {str(e)}"
    return templates.TemplateResponse("index1.html", {"request": request, "response": response})

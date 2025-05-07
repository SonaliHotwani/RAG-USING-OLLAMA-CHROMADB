from duckduckgo_search import DDGS
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# print("success")
llm = None
Settings.llm = Ollama(model="llama3.2:3b", request_timeout=360.0)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
Settings.embed_model = embed_model


def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract 2nd integer(b) from 1st integer(a) and returns the result integer"""
    return a - b


def search(query: str) -> str:
    """
    Args:
        query: user prompt
    return: context (str):
    search results to the user query
    """
    req = DDGS()
    response = req.text(query, max_results=3)
    context = ""
    for result in response:
        context += result['body']
    return context


search_tool = FunctionTool.from_defaults(fn=search)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
multiply_tool = FunctionTool.from_defaults(fn=multiply)

fntools = [multiply_tool, add_tool, subtract_tool, search_tool]

agent = ReActAgent.from_tools(fntools, llm=Settings.llm, max_iterations=15, verbose=True)

response = agent.chat("What is the distance between earth and moon plus 5000 kilometers")
print(response)

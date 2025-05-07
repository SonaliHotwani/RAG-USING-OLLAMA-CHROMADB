import streamlit as st
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


def convert_km_to_miles(km: float) -> float:
    """Convert kilometers to miles."""
    return km * 0.621371


search_tool = FunctionTool.from_defaults(fn=search)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
multiply_tool = FunctionTool.from_defaults(fn=multiply)
convert_km_to_miles_tool = FunctionTool.from_defaults(fn=convert_km_to_miles)

fntools = [multiply_tool, add_tool, subtract_tool, search_tool, convert_km_to_miles_tool]

st.title("AI-Powered Q&A Agent")
user_query = st.text_input("Ask a question:", "")

if st.button("Submit"):
    if user_query:
        agent = ReActAgent.from_tools(fntools, llm=Settings.llm, max_iterations=15, verbose=True)
        answer = agent.chat(user_query)
        st.write("Answer:", answer)
    else:
        st.warning("Please enter a question.")

st.subheader("Kilometer to Mile Converter")
km_value = st.slider("Select Kilometers", min_value=0.0, max_value=10000.0, step=0.1)
miles = convert_km_to_miles(km_value)
st.write(f"{km_value} kilometers is equal to {miles} miles.")

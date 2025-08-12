## Integrate our code with OpenAI API
import os


from api_key import openai_key
from langchain.llms import OpenAI
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key
# streamlit framework
st.title("Langchain Demo with OpenAI API")
input_text = st.text_input("Search the topic You want")

# OpenAI LLM models
llm = OpenAI(temeperature=0.8)

if input_text:
    response = llm(input_text)
    st.write(response)

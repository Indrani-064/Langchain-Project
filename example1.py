## Integrate our code with OpenAI API
import os

from constants import openai_key
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
os.environ["OPENAI_API_KEY"] = openai_key
# streamlit framework
st.title("Celebrity Search Results")
input_text = st.text_input("Search the topic You want")

#prompt template
first_input_prompt = PromptTemplate(
    input_variables=["name"],
    template="Tell me about celebrity, named {name}",
    )
# Memory

person_memory = ConversationBufferMemory(input_key="name", output_key="person", memory_key="chat_history")
dob_memory = ConversationBufferMemory(input_key="person", output_key="DOB", memory_key="chat_history")
description_memory = ConversationBufferMemory(input_key="DOB", output_key="description", memory_key="chat_history")

# OpenAI LLM models
llm = OpenAI(temperature=0.8)
chain = LLMChain(
    llm=llm,
    prompt=first_input_prompt,verbose=True, output_key="person", memory=person_memory 
)
second_input_prompt = PromptTemplate(
    input_variables=["person"],
    template="When the person born {person}?",
)

chain2 = LLMChain(
    llm=llm,
    prompt=second_input_prompt, verbose=True, output_key="DOB", memory=dob_memory
)
third_input_variables = PromptTemplate(input_variables=['DOB'],
                                       template="Mention 5 major events happened around {DOB} in the world")
chain3 = LLMChain(
    llm=llm,
    prompt=third_input_variables, verbose=True, output_key="description", memory=description_memory
)
parent_chain = SequentialChain(chains=[chain, chain2,chain3],input_variables=['name'],
                               output_variables=['person','DOB','description'] ,verbose=True)

if input_text:
    response = parent_chain({'name':input_text})
    st.write(response)

    with st.expander("Person name"):
        st.info(person_memory.buffer)
    with st.expander("Date of Birth"):
        st.info(dob_memory.buffer)
    with st.expander("Description"):
        st.info(description_memory.buffer)
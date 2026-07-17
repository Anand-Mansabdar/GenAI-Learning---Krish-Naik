import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv

load_dotenv()

# API Wrappers
arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=150)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=200)
wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)

search = DuckDuckGoSearchRun(name="Search")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter API Key:", type="password")

if "messages" not in st.session_state:
  st.session_state["messages"] = [
    {"role": "assistant",
     "content": "Search the web and chat with the assistant" 
    }
  ]
  
  for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])
    
  if prompt := st.chat_input(placeholder="Explain GenAI."):
    st.session_state.messages.append(
      {"role": "human", 
       "content": prompt
      })
    st.chat_message("user").write(prompt)
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, streaming=True)
    tools = [arxiv, wikipedia, search]
    
    search_agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_error=True)
    
    with st.chat_message("assistant"):
      callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
      response = search_agent.run(st.session_state.messages, callbacks=[callback])
      
      st.session_state.messages.append({
        "role": "assistant",
        "content": "response"
      })
      
      st.write(response)

    
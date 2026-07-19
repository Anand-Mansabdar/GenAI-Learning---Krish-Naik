import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_classic.chains import LLMChain, LLMMathChain
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents import Tool, initialize_agent
from langchain_classic.callbacks import StreamlitCallbackHandler


llm = ChatGroq(model="llama-3.3-70b-versatile")

wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia = Tool(
  name="Wikipedia",
  func=wikipedia_wrapper.run,
  description="Search the internet to find various information"
)

math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
  name="Calculator",
  func=math_chain.run,
  description="Tool for answering math related problems. Only input mathematical expression."
)

prompt = """  
  You are an expert agent in answering mathematical questions. Logically arrive at the solution and display it point-wise for the question below
  Question: {question}
  Answer:
"""

prompt_template = PromptTemplate(input_variables=["question"], template=prompt)


chain = LLMChain(llm=llm, prompt=prompt_template)


reasoning_tool = Tool(
  name="Reasoning Tool",
  func=chain.run,
  description="Tool for answering logic-based and reasoning questions"
)

assistant_agent = initialize_agent(
  tools=[reasoning_tool, wikipedia, calculator],
  llm=llm,
  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
  verbose=False,
  handle_parsing_errors=True
)

if "messages" not in st.session_state:
   st.session_state["messages"] = [
     {
       "role": "assistant",
       "content": "I am a Math Chatbot who can answer all your math related questions"
     }
   ]
   
for msg in st.session_state.messages:
  st.chat_message(msg["role"]).write(msg['content'])
  
  
def generate_response(question):
  response = assistant_agent.invoke({"question": question})
  return response

question_area = st.text_area("Enter your question: ")


if st.button("Find Solution"):
  if question_area:
    with st.spinner("Generating Response..."):
      st.session_state.messages.append({"role": "user", "content": question_area})
      st.chat_message("user").write(question_area)
      
      callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
      
      response = assistant_agent.run(st.session_state.messages, callbacks=[callback])
      
      st.session_state.messages.append({"role": "assistant", "content": response})
      
      st.write("Response:")
      st.success(response)
  else:
    st.warning("Enter the question...")
      
      
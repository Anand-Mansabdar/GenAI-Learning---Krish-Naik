import sqlite3
import streamlit as st
import pathlib as Path
from langchain_groq import ChatGroq
from langchain_classic.agents import create_sql_agent
from langchain_classic.sql_database import SQLDatabase
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_classic.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_classic.agents import initialize_agent, AgentType


st.title("Langchain Project: Chat with an SQL Database")

# 
INFECTION_WARNING = """
  SQL
"""

LOCAL_DB = "USE_LOCALDB"
MYSQL = "MY_SQL"

radio_options = ["Use SQLITE3 Database - Student.db", "Connect to your own local SQL Database"]

selected_option = st.sidebar.radio(label="Choose the Database", options=radio_options)

if radio_options.index(selected_option) == 1:
  db_uri = MYSQL
  mysql_host = st.sidebar.text_input("Provide MySQL host")
  mysql_user = st.sidebar.text_input("Enter MySQL User")
  mysql_password = st.sidebar.text_input("Enter MySQL Password", type="password")
  mysql_db = st.sidebar.text_input("Enter MySQL Database Name")
else:
  db_uri = LOCAL_DB

api_key = st.sidebar.text_input("Enter GROQ API Key", type="password")


if not db_uri:
  st.info("Enter the required details and try again")

if not api_key:
  st.info("Enter the GROQ API Key")


llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(db_uri, db_host=None, db_user=None, db_password=None, mysql_db=None):
  if db_uri == LOCAL_DB:
    file_path = (Path(__file__).parent/"student.db").absolute()
    creator = lambda: sqlite3.connect(f"file:{file_path}?mode=ro", uri=True)
    return SQLDatabase(create_engine("sqlite:///", creator=creator))
  elif db_uri == MYSQL:
    if not (db_host and db_user and db_password and mysql_db):
      st.error("Provide the required details...")
      st.stop()
    return SQLDatabase(create_engine(f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{mysql_db}"))
  
if db_uri == MYSQL:
  db = configure_db(db_uri, db_host=mysql_host,db_user=mysql_user, db_password=mysql_password, mysql_db=mysql_db)
else:
  db = configure_db(db_uri)
  
  
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
  llm=llm, toolkit=toolkit, verbose=True,
  agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
  st.session_state["messages"] = [{
    "role": "assistant",
    "content": "How can I help you?"
  }]

for msg in st.session_state.messages:
  st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything regarding the database...")

if user_query:
  st.session_state.messages.append({"role": "user", "content": user_query})
  st.chat_message("user").write(user_query)
  
  with st.chat_message("assistant"):
    callback = StreamlitCallbackHandler(st.container())
    response = agent.run(user_query, callbacks=[callback])
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.write(response)
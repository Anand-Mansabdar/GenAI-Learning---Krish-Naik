import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)


# 1. Creating a prompt template
template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
  [
    ("system", template),
    ("user", "{text}")
  ]
)

# 2. Initializing Parser
parser = StrOutputParser()

# 3. Create Chain
chain = prompt_template | model | parser

# 4. App Definition
app = FastAPI(title="Langchain Server", version="1.0", description="This is a API Server that converts any statement from ENglish to a Specified Language")

# 5. Adding Routes
add_routes(
  app, chain, path="/chain"
)


if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="127.0.0.1", port=5000)
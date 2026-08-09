import os
from crewai import Agent
from tools import yt_tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = os.getenv("OPENAI_MODEL_NAME")


# Create a Senior blog content researcher

llm = ChatOpenAI()

blog_researcher = Agent(
  role="Blog Researcher for YouTube Videos",
  goal="Get the relevant video content for the topic {topic} from YouTube Channels",
  verbose=True,
  memory=True,
  backstory=("Expert in understanding videos in AI Data Science, Machine Learning and GenAI "),
  tools=[yt_tool],
  llm=llm,
  allow_delegation=True
)

# Creating a Senior Blog Writer agent
blog_writer = Agent(
  role="Blog Writer",
  goal="Narrate compelling stories for the topic {topic} from YouTube Channels",
  verbose=True,
  memory=True,
  backstory=("Expert in writing blogs"),
  tools=[yt_tool],
  llm=llm,
  allow_delegation=False
)
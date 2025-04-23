from langchain_community.llms import OpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_xai import ChatXAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='/home/nareg/Desktop/paper_idea/llm_echo_chamber/.env')

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
x_api_key = os.getenv("X_API_KEY")

# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------

llm_gemini = None

llm_x = ChatXAI(
    xai_api_key=x_api_key,
    model="grok-2-1212",
)

llm_haiku = ChatAnthropic(
    model="claude-3-5-haiku-latest",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    api_key = anthropic_api_key
)


llm_sonnet = ChatAnthropic(
    model="claude-3-7-sonnet-latest",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    api_key = anthropic_api_key
)

llm_gpt4o = ChatOpenAI(openai_api_key = openai_api_key,
                temperature = 0.0,
                model = 'gpt-4o')

models = {'gemini': llm_gemini,
                'haiku': llm_haiku,
                'xAI': llm_x,
                'sonnet': llm_sonnet,
                'gpt-4o': llm_gpt4o}
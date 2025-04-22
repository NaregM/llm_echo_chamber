from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.output_parsers import PydanticOutputParser

from langchain_anthropic import ChatAnthropic


from dotenv import load_dotenv
import os

load_dotenv()

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------

llm_gemini = None
llm_haiku = None
llm_x = None
llm_gpt4o = None

llm_sonnet = ChatAnthropic(
    model="claude-3-7-sonnet-latest",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    api_key = anthropic_api_key
)

models = {'gemini': llm_gemini,
                'haiku': llm_haiku,
                'xAI': llm_x,
                'sonnet': llm_sonnet,
                'gpt-4o': llm_gpt4o}
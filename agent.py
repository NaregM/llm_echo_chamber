import numpy as np
from pydantic import BaseModel, field_validator, Field

import dotenv
from dotenv import find_dotenv, load_dotenv

import openai
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_anthropic import ChatAnthropic

import os
import re
import pickle

from prompts import Q_PROMPT


class QuestionModel(BaseModel):
    """
    """
    llm_answer: str = Field(description="4-5 sentece answer to the provided question.")

    @field_validator("llm_answer")
    def answer_len(cls, field):
        if len(str(field)) > 2_500:
            raise ValueError("too long!")
        return field
    
parser = PydanticOutputParser(pydantic_object=QuestionModel)

prompt = PromptTemplate(
    template=Q_PROMPT,
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


def llm_response(llm_model, question, prompt, parser):
    """
    """
    chain = prompt | llm_model | parser
    
    try:
        
        return chain.invoke({"question": question})
    
    except:
        
        return "Failed"
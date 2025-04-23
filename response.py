import numpy as np

import os
from typing import Dict, List
from tqdm import tqdm

from langchain_core.language_models.base import BaseLanguageModel
from agent import prompt, parser, llm_chian
from llm_models import models
from questions import questions

from models import LLMResponse
import pickle


# ---------------------------------------------------------
# ---------------------------------------------------------
# ---------------------------------------------------------

def llm_response(model: BaseLanguageModel, questions: Dict[str, str]) -> List[Dict[str, str]]:
    
    """
    """
    llm_resps = []
    if not model:
        return None
    
    topic = list(questions.keys())[0]
    qs = list(questions.values())[0]
    
    for question in qs:
        
        response_obj = LLMResponse(question=question,
                                   llm_answer=llm_chian(model, question,
                                                   prompt, parser).llm_answer,
                                   topic=topic)
        
        llm_resps.append(response_obj.model_dump())
        
    return llm_resps

questions2 = {
    "Culture": [
        "How does cultural heritage influence modern societal values?",
        "In what ways does art preserve traditional customs?",
        "How do cultural festivals contribute to community identity?",
        "What role does language play in shaping cultural identity?",
        "How is globalization affecting local cultural practices?",
        "How does cultural appropriation impact society?",
        "What is the influence of cultural icons on youth?",
        "How do multicultural societies navigate cultural differences?",
        "What is the role of museums in preserving cultural history?",
        "How do cultural narratives shape national identity?"
    ]}

def save_model_response(model_name: str, response, output_dir: str = "results"):
    """
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{model_name}_response.pkl")
    
    with open(file_path, "wb") as f:
        
        pickle.dump(response, f)
        
    print(f"Saved response for {model_name} to {file_path}")

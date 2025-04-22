import numpy as np
from agent import prompt, parser, llm_response
from llm_models import models
from questions import questions

llm_resps = {}

for model_k, model_v in models.items():
    
    res_tmp = {}
    
    for topic, question in questions.items():
    
        res_tmp[topic+';'+question] = llm_response(model_v, question,
                                                   prompt, parser)
        
    llm_resps[model_k] = res_tmp
    
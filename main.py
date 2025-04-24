from llm_models import models
from response import save_model_response, questions2, llm_response
from pairwise_jsd import compute_tau_q
from tqdm import tqdm
import pickle


if __name__ == "__main__":
    
    with open("/home/nareg/Desktop/paper_idea/llm_echo_chamber/results/gpt-4o_response.pkl", "rb") as f:
        gpt_res = pickle.load(f)
        
    with open("/home/nareg/Desktop/paper_idea/llm_echo_chamber/results/sonnet_response.pkl", "rb") as f:
        sonnet_res = pickle.load(f)
        
    with open("/home/nareg/Desktop/paper_idea/llm_echo_chamber/results/haiku_response.pkl", "rb") as f:
        haiku_res = pickle.load(f)
        
    with open("/home/nareg/Desktop/paper_idea/llm_echo_chamber/results/xAI_response.pkl", "rb") as f:
        x_res = pickle.load(f)
        
    responses_by_llm = {'gpt4o': gpt_res,
                        'sonnet': sonnet_res,
                        'haiku': haiku_res,
                        'xAI': x_res}
    
    qs = list(questions2.values())[0]
    
    print(f'tau_q: {compute_tau_q(responses_by_llm, qs)}')
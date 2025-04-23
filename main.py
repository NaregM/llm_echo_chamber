from llm_models import models
from response import save_model_response, questions2, llm_response
from tqdm import tqdm


if __name__ == "__main__":
    
    for name, model in tqdm(models.items()):
        response = llm_response(model, questions2)
        save_model_response(name, response)

from pydantic import BaseModel

class LLMResponse(BaseModel):
    question: str
    topic: str
    llm_answer: str
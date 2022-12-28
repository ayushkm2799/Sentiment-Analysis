

from pydantic import BaseModel

class sentiment_request(BaseModel):
    text: str
    model_type: str



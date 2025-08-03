from pydantic import BaseModel

class SaveMessageRequest(BaseModel):
    chat_id: str
    question: str
    response: str


class AskRequest(BaseModel):
    diagnosis: str
    question: str


class JustAskRequest(BaseModel):
    question: str

from fastapi import FastAPI
from pydantic import BaseModel
from app import get_and_send
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Or specify your Flutter app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReqData(BaseModel):
    name:str

class Query(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    generatedQuery: str
    result: str
    success: bool


@app.post("/query/")
async def query(query: Query):
    cleaned_sql, result= get_and_send(query.question)

    return QueryResponse(
        question=query.question,
        generatedQuery=cleaned_sql,
        result=result,
        success=True,
    )

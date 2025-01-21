from fastapi import FastAPI
from pipeline.query_handler import handle_query
from pydantic import BaseModel

app = FastAPI()

class UserRequest(BaseModel):
    user_message: str

# health check
@app.get('/')
async def health_check():
    return {"status": "ok"}

@app.post("/query")
async def query_pipeline(request: UserRequest):
    
    user_message = request.user_message
    response = await handle_query(user_message)
    return {"response": response}
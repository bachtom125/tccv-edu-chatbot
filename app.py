from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
# from firebase.firebase_auth import verify_firebase_token
from fastapi.responses import StreamingResponse
from pipeline.langchain_bot import full_workflow
from pipeline.utils import parse_input

app = FastAPI()

class UserRequest(BaseModel):
    user_message: str

# Health check
@app.get('/')
async def health_check():
    return {"status": "ok"}

@app.post("/api/send-message")
async def stream_response(user_input: dict):
    past_messages, user_query = parse_input(user_input["messages"])
    
    # Create input for full chain
    chain_inputs = {"user_query": user_query, "past_messages": past_messages}

    # ✅ Directly return the async generator
    return StreamingResponse(full_workflow(user_query, past_messages), media_type="text/plain")
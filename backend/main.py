from fastapi import FastAPI, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from note_model import NoteModel
import json
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init Path
DATA_FILE = "data.json"

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as file:
        json.dump([], file)

class NoteIn(BaseModel):
    text: str

class NoteOut(BaseModel):
    id: int
    text: str

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_text_to_all(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_text_to_all(data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/submit_note")
async def submit_note(
    text: str = Form(...)
):
    print(f"Received note: {text}")

    model_path = "gpt2"  # GPT-2 model identifier
    model = NoteModel(model_id_or_path=model_path, optimize=True)
    
    raw_output = model.generate_output(text)
    formatted_output = model.format_structured_output(raw_output)


    data = {
        "id": None,  # Placeholder for ID, will be set later
        "text": formatted_output
    }

    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as file:
            existing_data = json.load(file)
    else:
        existing_data = []

    new_id = len(existing_data) + 1
    data["id"] = new_id
    existing_data.append(data)

    with open(DATA_FILE, "w") as file:
        json.dump(existing_data, file, indent=4)

    return {"message": "Note saved successfully"}

@app.get("/notes", response_model=List[NoteOut])
async def get_notes():
    with open(DATA_FILE, "r") as file:
        notes = json.load(file)
    return notes

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

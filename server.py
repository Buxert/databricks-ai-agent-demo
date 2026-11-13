import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
from agent import run_agent

app = FastAPI()

class ChatRequest(BaseModel):
    input: list
    thread_id: Optional[str] = None

@app.get("/")
async def root():
    return HTMLResponse(Path("chat.html").read_text())

@app.get("/rockstars_logo.png")
async def logo():
    p = Path("rockstars_logo.png")
    if p.exists():
        return FileResponse(p, media_type="image/png")
    return HTMLResponse("", status_code=404)

@app.post("/invocations")
async def invoke(req: ChatRequest, request: Request):
    raw_user = request.headers.get("X-Forwarded-Preferred-Username", "anonymous")
    user_id = raw_user.replace(".", "_").replace("@", "_at_")
    messages = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in req.input]
    try:
        result = await run_agent(messages, thread_id=req.thread_id, user_id=user_id)
        # run_agent returns {"output": ..., "thread_id": ...}
        if isinstance(result, dict) and "output" in result:
            return result
        return {"thread_id": req.thread_id, "output": result}
    except Exception:
        # Temporary debug surface: show the real error in the chat bubble
        import traceback
        return {"thread_id": req.thread_id,
                "output": "\u26a0\ufe0f DEBUG - the agent failed with:\n\n" + traceback.format_exc()[-1800:]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

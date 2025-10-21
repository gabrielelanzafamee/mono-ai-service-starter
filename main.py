from dotenv import load_dotenv

import os
import uvicorn

from fastapi import FastAPI

load_dotenv()

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=os.getenv("PORT"))
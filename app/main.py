from fastapi import FastAPI
from app.routers.mnist_router import router


app = FastAPI()
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Hello World"}


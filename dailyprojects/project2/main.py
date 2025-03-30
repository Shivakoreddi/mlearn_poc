from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message":"Hello,FastAPI"}

@app.get("/greet/")
async def greet_path():
    return f"Hi,FastAPI"

@app.get("/greet/item")
async def greet_item():
    return f"Hi, This is item page!"

@app.get("/greet/all")
async def greet_item():
    return f"Hi, all, welcome to api developments!"

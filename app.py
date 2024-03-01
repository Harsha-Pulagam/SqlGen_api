from fastapi import FastAPI, APIRouter
import uvicorn
from inference import Inference
from model import Model
import logging

logging.basicConfig(level = logging.INFO)

app = FastAPI()
router = APIRouter()
inference = Inference()

@router.get("/")
async def home():
  return {"message": "Machine Learning service"}

@router.get("/metadata")
async def get_metadata():
  try:
    with open("metadata.sql", "r") as file:
      metadata = file.read()
    return {"metadata": metadata}
  except Exception as e:
    log.error("Failed to retrieve metadata")

@router.put("/metadata")
async def post_metadata(metadata: dict):
  try:
    text = metadata["metadata"]
    with open("metadata.sql", "w") as file:
      file.write(text)
    return {"message": "Metadata updated successfully"}
  except Exception as e:
    log.error("Failed to update metadata")

@router.post("/sqlcoder")
async def data(data: dict):
  try:
    input_text = data["text"]
    res = inference.generate_query(input_text)
    return res
  except Exception as e:
    log.error("Something went wrong")

app.include_router(router)

if __name__ == "__main__":
  uvicorn.run("app:app", reload=True,port =8051,host="0.0.0.0")


from fastapi import FastAPI, APIRouter
import uvicorn
from inference import Inference
from model import Model
from sqlcoder import Sqlcoder
import logging

logging.basicConfig(level = logging.INFO)

app = FastAPI()
sql = Sqlcoder()
router = APIRouter()
inference = Inference()

@router.get("/")
async def home():
  return {"message": "Machine Learning service"}

@router.post("/sqlcoder")
async def data(data: dict):
  try:
    input_text = data["text"]
    res = sql.sql_generator(input_text, inference)
    return res
  except Exception as e:
    log.error("Something went wrong")

app.include_router(router)

if __name__ == "__main__":
  uvicorn.run("app:app", reload=True,port =8051,host="0.0.0.0")
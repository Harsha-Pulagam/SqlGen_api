from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Model:
  """A model class to load the model and tokenizer"""

  def __init__(self) -> None:
    pass
  
  def load_model():
    model = AutoModelForCausalLM.from_pretrained("/workspace/ml-service/model",
                                                 torch_dtype=torch.float16,
                                                 device_map ="auto")
    return model

  def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("/workspace/ml-service/model")
    return tokenizer
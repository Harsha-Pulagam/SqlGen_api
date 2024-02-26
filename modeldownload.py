from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "defog/sqlcoder-7b-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=0
    )

save_dir = "/workspace/ml-service/model"

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
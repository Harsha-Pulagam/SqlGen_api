from model import Model
import sqlparse
import torch

class Inference:
    def __init__(self):
        self.model = Model.load_model()
        self.tokenizer = Model.load_tokenizer()
        
    def generate_prompt(self, question, prompt_file="prompt.md", metadata_file="metadata.sql"):
        with open(prompt_file, "r") as f:
            prompt = f.read()
        
        with open(metadata_file, "r") as f:
            table_metadata_string = f.read()

        prompt = prompt.format(
            user_question=question, table_metadata_string=table_metadata_string
        )
        return prompt
    
    def generate_query(self, question):
        updated_prompt = self.generate_prompt(question)
        inputs = self.tokenizer(updated_prompt, return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(
            **inputs,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=400,
            do_sample=False,
            num_beams=1,
        )
        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # empty cache so that you do generate more results w/o memory crashing
        # particularly important on Colab – memory management is much more straightforward
        # when running on an inference service
        return sqlparse.format(outputs[0].split("[SQL]")[-1], reindent=True)
     
        
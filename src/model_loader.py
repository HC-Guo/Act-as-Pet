import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelLoader:
    def __init__(self, model_url):
        self.tokenizer = AutoTokenizer.from_pretrained(model_url, trust_remote=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_url).to(self.device)

    
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
    

if __name__ == "__main__":
    ModelLoader


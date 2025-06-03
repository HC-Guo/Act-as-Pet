import json
import tqdm
from pathlib import Path
from .metrics import cal_bleu, cal_rouge, get_embeddings_score, cal_intent_acc
from .config import load_config
from .model_loader import ModelLoader
from .llm_eval import eval_llm_score

class Evaluator:
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        self.config = load_config()
        self.data_root_path = Path(self.config["data_root_path"])
        self.data_name_list = self.config['data_name_list']

    def load_dataset(self, data_name: str):
        with open(self.data_root_path / f"{data_name}.json", 'r') as f:
            data = json.load(f)
        return data
    
    def generate_results(self, dataset, data_name):
        results = []
        
        bleu1_scores = []
        bleu2_scores = []
        bleu3_scores = []
        bleu4_scores = []

        rouge1_scores = []
        rougeL_scores = []
        embeddings_scores = []
        acc_scores = []
        llm_scores = []

        model = self.model_loader.get_model()
        tokenizer = self.model_loader.get_tokenizer() 
        for item in tqdm.tqdm(dataset, desc="inference"):
            prompt, label = item['prompt'], item['label']
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_input = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_input,
                max_new_tokens=512,
            )

            generated_ids = [
                output_ids[len(input_ids): ] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            bleu_score = cal_bleu([label], response)
            rouge_score = cal_rouge([label], response)
            emb_score = get_embeddings_score(label, response)

            bleu1_scores.append(bleu_score['bleu-1'])
            bleu2_scores.append(bleu_score['bleu-2'])
            bleu3_scores.append(bleu_score['bleu-3'])
            bleu4_scores.append(bleu_score['bleu-4'])
            rouge1_scores.append(rouge_score['rouge1'])
            rougeL_scores.append(rouge_score['rougeL'])
            embeddings_scores.append(emb_score)

            if data_name == 'PetNote_Intent_Recognition':
                acc_scores.append(cal_intent_acc(label, response))

            if data_name in ["PetRoutine_Level_I", "PetRoutine_Level_II", "PetRoutine_Level_III"]:
                llm_score = eval_llm_score(prompt, response, data_name)
                if llm_score is not None:
                    llm_scores.append(llm_score)


        bleu1_score = sum(bleu1_scores) / len(bleu1_scores)
        bleu2_score = sum(bleu2_scores) / len(bleu2_scores)
        bleu3_score = sum(bleu3_scores) / len(bleu3_scores)
        bleu4_score = sum(bleu4_scores) / len(bleu4_scores)
        rouge1_score = sum(rouge1_scores) / len(rouge1_scores)
        rougeL_score = sum(rougeL_scores) / len(rougeL_scores)
        embeddings_score = sum(embeddings_scores) / len(embeddings_scores)
        

        if data_name == self.config['cal_acc_data_name']:
            acc_scores = sum(acc_scores) / len(acc_scores)
            avg_score = (bleu1_score + bleu2_score + bleu3_score + bleu4_score + rouge1_score + rougeL_score + embeddings_score + acc_scores) / 8
        
        elif data_name in ["PetRoutine_Level_I", "PetRoutine_Level_II", "PetRoutine_Level_III"]:
            if llm_scores:
                llm_score = sum(llm_scores) / len(llm_scores)
            else:
                llm_score = 0
            avg_score = (bleu1_score + bleu2_score + bleu3_score + bleu4_score + rouge1_score + rougeL_score + embeddings_score + llm_score) / 8

        else:
            avg_score = (bleu1_score + bleu2_score + bleu3_score + bleu4_score + rouge1_score + rougeL_score + embeddings_score) / 7

        return avg_score
    
    def run(self):
        results = {}
        total_score = 0
        for data_name in self.data_name_list:
            dataset = self.load_dataset(data_name)
            
            score = self.generate_results(dataset, data_name)
            total_score += score

            print("Data: ", data_name)
            print("Score: ", score)
        
        total_score = total_score / len(self.data_name_list)
        print("Total Score: ", total_score)
        return results
    
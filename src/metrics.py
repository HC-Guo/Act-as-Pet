import warnings
import torchvision
warnings.filterwarnings("ignore", category=UserWarning)
torchvision.disable_beta_transforms_warning()
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from tqdm import tqdm
import torch
import re
from.config import load_config


config = load_config()

nltk.data.path.append(config["nltk_data_path"])

device = "cuda" if torch.cuda.is_available() else "cpu"

emb_model = SentenceTransformer(config["bge_path"]).to(device)

def get_embeddings_score(ref:str="test", hyp:str="test"):
    ref_emb = emb_model.encode([ref], convert_to_tensor=True, normalize_embeddings=True).to('cpu')
    hyp_emb = emb_model.encode([hyp], convert_to_tensor=True, normalize_embeddings=True).to('cpu')

    cos_sim = torch.nn.functional.cosine_similarity(ref_emb, hyp_emb).item()

    return cos_sim


def tokenize(text):
    return word_tokenize(text)


def cal_bleu(ref: list, candidate: str):
    ref = [tokenize(r) for r in ref]
    candidate = tokenize(candidate)
    bleu_score = {}
    total_score = 0
    for n in range(1, 5):
        weights = [0 for _ in range(1,5)]
        weights[n-1] = 1
        score = sentence_bleu(ref, candidate, weights=weights)
        bleu_score[f'bleu-{n}'] = score
        total_score += score
    
    bleu_score['bleu'] = total_score / 4

    return bleu_score
    

def cal_rouge(ref: list, candidate: str):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(r, candidate) for r in ref]
    # print(scores)
    avg_scores = {}
    for key in scores[0].keys():
        # print([score[key] for score in scores])
        avg_scores[key] = sum([score[key].fmeasure for score in scores]) / len(scores)
    
    return avg_scores

def get_intent(candidate:str):
    pattern = r'(?:Intention|Intent): (.*?)\nQuery:'
    match = re.search(pattern, candidate)
    if match:
        intention = match.group()
        if intention is not None:
            if 'Active'.lower() in intention.lower():
                return 'active'
            if 'Direct'.lower() in intention.lower():
                return 'direct'
            if 'Topic'.lower() in intention.lower():
                return 'topic'
    return None

def cal_intent_acc(ref: str, candidate: str):
    print(candidate)
    candidate_type = get_intent(candidate)
    print(candidate_type)
    print(ref)
    # print(ref.lower().count(candidate_type))
    if candidate_type and ref.lower().count(candidate_type) > 0:
        return 1
    
    return 0

if __name__ == "__main__":
    print(cal_intent_acc("active", "Intention: Active\nQuery: test"))


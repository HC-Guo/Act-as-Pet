from src.model_loader import ModelLoader
from src.evaluator import Evaluator
from src.config import load_config


if __name__ == "__main__":
    config = load_config()
    model_loader = ModelLoader(config["model_url"])
    evaluator = Evaluator(model_loader)
    evaluator.run()
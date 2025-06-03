### Act-as-Pet

### Instructions for Use
- Download the data to the `data/` directory.
- Download the `bge-base-en-v1.5` model and place it in the specified location.
- Download the nltk tokenizer resource package.
- Download the model to be evaluated.
- Configure `config.yaml`.
- For the LLM evaluation part, implement the `curl_llm` function in `src/llm_eval.py` to access the LLM for evaluation. The default implementation uses an API key to access the API. 
    - If the configuration of url remains xxxx, it will directly return 0.
- Run `python run.py`.

### Partial Dependency Packages
- `torch==2.6.0`
- `tqdm`
- `sentence_transformers`
- `nltk`
- `tenacity`

### Field Descriptions
- `nltk_data_path`: Local path to the nltk resource package.
- `bge_path`: Path to the BGE scoring model, used for embedding-based scoring.
- `model_url`: Model path.
- `data_root_path`: Data path.
- `data_name_list`: List of datasets to be evaluated.
- `eval_llm_url`: LLM URL passed to the `curl_llm` function.
- `eval_llm_api_key`: Corresponding API key.

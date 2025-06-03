import json
import requests
import re
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential
from .config import load_config

config = load_config()

def before_sleep_callback(retry_state):
    print(f"Retrying... {retry_state}")

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2), before_sleep=before_sleep_callback)
def curl_llm(user_chat="", system_prompt="You are a helpful assistant"):
    url = config['eval_llm_url']
    if url == "xxxxx":
        print("Please provide the url of the LLM evaluation")
        return "score: 0.0"
    
    headers = {
        'api-key': config['eval_llm_api_key'],
        'Content-Type': 'application/json'
    }
    data = {
        'messages': [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': user_chat
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    res_dict = response.json()

    res_content = res_dict['messages'][0]['message']['content']

    return res_content


def get_pet_routine_level_1(text):
    pattern = re.compile(r'Now, you need generate the output:\n(.*?)\nOutput:', re.DOTALL)
    ans = re.search(pattern, text).group()
    return ans

def routine_level_1_func(item, user_prompt_template):
    setting = get_pet_routine_level_1(item['prompt'])
    user_prompt = user_prompt_template.format(
        setting=setting,
        output=item['output']
    )
    return user_prompt


def get_pet_routine_level_2(text):
    pattern = re.compile(r'Now, you need generate the output:\n(.*?)\nOutput:', re.DOTALL)
    ans = re.search(pattern, text).group()
    return ans

def routine_level_2_func(item, user_prompt_template):
    setting = get_pet_routine_level_2(item['prompt'])
    user_prompt = user_prompt_template.format(
        setting=setting,
        output=item['output']
    )
    return user_prompt

def get_pet_routine_level_3(text):
    pattern = re.compile(r'Now, you need generate the output:\n(.*?)\nOutput:', re.DOTALL)
    ans = re.search(pattern, text).group()
    return ans

def routine_level_3_func(item, user_prompt_template):
    setting = get_pet_routine_level_3(item['prompt'])
    user_prompt = user_prompt_template.format(
        setting=setting,
        output=item['output']
    )
    return user_prompt

def get_score(text):
    pattern = r'\d+'

    try:
        match = re.search(pattern, text)
        print(f"match: {match}")
        if match:
            score = int(match.group())
            if 0 <= score <= 5:
                return score / 5
            else:
                return None
            
    except:
        return None
    

def eval_llm_score(prompt, ans, type="PetRoutine_Level_I"):
    assert type in ["PetRoutine_Level_I", "PetRoutine_Level_II", "PetRoutine_Level_III"]
    with open(f"prompt/{type}.txt", 'r') as f:
        user_prompt_template = f.read()
    
    if type == "PetRoutine_Level_I":
        user_prompt = routine_level_1_func({'prompt': prompt, 'output': ans}, user_prompt_template)
    elif type == "PetRoutine_Level_II":
        user_prompt = routine_level_2_func({'prompt': prompt, 'output': ans}, user_prompt_template)
    elif type == "PetRoutine_Level_III":
        user_prompt = routine_level_3_func({'prompt': prompt, 'output': ans}, user_prompt_template)
    else:
        assert "Invalid type, type need be one of ['PetRoutine_Level_I', 'PetRoutine_Level_II', 'PetRoutine_Level_III']"
    res = curl_llm(user_chat=user_prompt)
    print(f"res: {res}")
    score = get_score(res)


    return score



        


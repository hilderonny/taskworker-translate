PROGRAM_VERSION = '1.0.0'

print(f'Translator Version {PROGRAM_VERSION}')

import time
import os
import json
import requests
import datetime

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--apiurl', type=str, action='store', required=True, help='Root URL of the API of the task bridge to use, e.g. https://taskbridge.ai/api/')
parser.add_argument('--version', '-v', action='version', version=PROGRAM_VERSION)
args = parser.parse_args()

import os
APIURL = args.apiurl
if not APIURL.endswith("/"):
    APIURL = f"{APIURL}/"
print(f'Using API URL {APIURL}')

# Load AI
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B").to(device)
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")

def check_and_process():
    start_time = datetime.datetime.now()
    req = requests.get(f"{APIURL}take/translate")
    task = req.json()
    print(json.dumps(task, indent=2))
    if req.status_code != 200:
        print(task["error"])
        return False
    sourcelanguage = task["data"]["sourcelanguage"]
    targetlanguage = task["data"]["targetlanguage"]
    texttotranslate = task["data"]["text"]
    result_to_report = {}

    tokenizer.src_lang = sourcelanguage
    encoded_hi = tokenizer(texttotranslate, return_tensors="pt").to(device)
    generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id(targetlanguage))
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    result_to_report["text"] = result
    end_time = datetime.datetime.now()
    result_to_report["duration"] = (end_time - start_time).total_seconds()
    print(json.dumps(result_to_report, indent=2))
    print('Reporting result')
    requests.post(f"{APIURL}finish/{task["id"]}/", json=result_to_report)
    return True

try:
    print('Ready and waiting for action')
    while True:
        text_was_processed = False
        try:
            text_was_processed = check_and_process()
        except Exception as ex:
            print(ex)
        if text_was_processed == False:
            time.sleep(3)
except Exception:
    pass

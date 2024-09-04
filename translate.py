REPOSITORY = "https://github.com/hilderonny/taskworker-translate"
VERSION = '1.1.0'
LIBRARY = "transformers"
MODEL = "facebook/m2m100_1.2B"
DEVICE = "cuda:0"

print(f'Translator Version {VERSION}')

import time
import os
import json
import requests
import datetime

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--apiurl', type=str, action='store', required=True, help='Root URL of the API of the task bridge to use, e.g. https://taskbridge.ai/api/')
parser.add_argument('--version', '-v', action='version', version=VERSION)
parser.add_argument('--sourcelanguage', action='store', required=True, help='The source language the worker should be able to process')
parser.add_argument('--targetlanguage', action='store', required=True, help='The target language the worker should output')
args = parser.parse_args()

import os
APIURL = args.apiurl
if not APIURL.endswith("/"):
    APIURL = f"{APIURL}/"
print(f'Using API URL {APIURL}')
SOURCELANGUAGE = args.sourcelanguage
print(f'Using source language {SOURCELANGUAGE}')
TARGETLANGUAGE = args.targetlanguage
print(f'Using target language {TARGETLANGUAGE}')

# Load AI
import torch
if not torch.cuda.is_available():
    DEVICE = "cpu"
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
transformer_model = M2M100ForConditionalGeneration.from_pretrained(MODEL).to(DEVICE)
tokenizer = M2M100Tokenizer.from_pretrained(MODEL)
tokenizer.src_lang = SOURCELANGUAGE
token_id = tokenizer.get_lang_id(TARGETLANGUAGE)

def check_and_process():
    start_time = datetime.datetime.now()
    take_request = {}
    take_request["type"] = "translate"
    take_request["abilities"] = {}
    take_request["abilities"]["sourcelanguage"] = SOURCELANGUAGE
    take_request["abilities"]["targetlanguage"] = TARGETLANGUAGE
    req = requests.post(f"{APIURL}tasks/take/", json=take_request)
    if req.status_code != 200:
        return False
    task = req.json()
    taskid = task["id"]
    print(json.dumps(task, indent=2))
    textstotranslate = task["data"]["texts"]
    result_to_report = {}
    result_to_report["result"] = {}
    result_to_report["result"]["texts"] = []

    for text in textstotranslate:
        if len(text) < 1:
            result_to_report["result"]["texts"].append("")
            continue
        encoded = tokenizer(text, return_tensors="pt").to(DEVICE)
        generated_tokens = transformer_model.generate(**encoded, forced_bos_token_id=token_id)
        result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        result_to_report["result"]["texts"].append("".join(result))
    end_time = datetime.datetime.now()
    result_to_report["result"]["device"] = DEVICE
    result_to_report["result"]["duration"] = (end_time - start_time).total_seconds()
    result_to_report["result"]["repository"] = REPOSITORY
    result_to_report["result"]["version"] = VERSION
    result_to_report["result"]["library"] = LIBRARY
    result_to_report["result"]["model"] = MODEL
    print(json.dumps(result_to_report, indent=2))
    print("Reporting result")
    requests.post(f"{APIURL}tasks/complete/{taskid}/", json=result_to_report)
    print("Done")
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

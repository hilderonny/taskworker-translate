from importlib.metadata import version
import time
import json
import requests
import datetime
import argparse
import torch
import os
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langdetect import detect

REPOSITORY = "https://github.com/hilderonny/taskworker-translate"
VERSION = "1.2.0"
LIBRARY = "transformers-" + version("transformers")
MODEL = "facebook/m2m100_1.2B"
DEVICE = "cuda:0"
APIVERSION = "v2"
LOCAL_MODEL_PATH = "./models/facebook/m2m100_1.2B"

print(f'Translator Version {VERSION}')

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--taskbridgeurl', type=str, action='store', required=True, help='Root URL of the API of the task bridge to use, e.g. https://taskbridge.ai/')
parser.add_argument('--version', '-v', action='version', version=VERSION)
parser.add_argument('--worker', type=str, action='store', required=True, help='Unique name of this worker')
args = parser.parse_args()

WORKER = args.worker
print(f'Worker name: {WORKER}')
TASKBRIDGEURL = args.taskbridgeurl
if not TASKBRIDGEURL.endswith("/"):
    TASKBRIDGEURL = f"{TASKBRIDGEURL}/"
APIURL = f"{TASKBRIDGEURL}api/{APIVERSION}/"
print(f'Using API URL {APIURL}')

# Load AI
if not torch.cuda.is_available():
    DEVICE = "cpu"
# Save online model locally. Only needed once.
if not os.path.exists(LOCAL_MODEL_PATH):
    M2M100ForConditionalGeneration.from_pretrained(MODEL).save_pretrained(LOCAL_MODEL_PATH)
transformer_model = M2M100ForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH).to(DEVICE)
tokenizer = M2M100Tokenizer.from_pretrained(MODEL)

def check_and_process():
    start_time = datetime.datetime.now()
    take_request = {}
    take_request["type"] = "translate"
    take_request["worker"] = WORKER
    req = requests.post(f"{APIURL}tasks/take/", json=take_request)
    if req.status_code != 200:
        return False
    task = req.json()
    taskid = task["id"]
    print(json.dumps(task, indent=2))
    source_language = None
    if "sourcelanguage" in task["data"]:
        source_language = task["data"]["sourcelanguage"]
    target_language = task["data"]["targetlanguage"]
    texts_to_translate = task["data"]["texts"]
    result_to_report = {}
    result_to_report["result"] = {}
    result_to_report["result"]["texts"] = []
    token_id = tokenizer.get_lang_id(target_language)

    for text_to_translate in texts_to_translate:
        result_element = {}
        if len(text_to_translate) < 1:
            result_element["text"] = ""
        else:
            try:
                # When a language cannot be translated, return the original text
                if source_language is None:
                    detected_language = detect(text_to_translate)[:2] # Only the first two digits
                else:
                    detected_language = source_language
                result_element["sourcelanguage"] = detected_language
                print(text_to_translate, detected_language)
                tokenizer.src_lang = detected_language
                encoded = tokenizer(text_to_translate, return_tensors="pt").to(DEVICE)
                generated_tokens = transformer_model.generate(**encoded, forced_bos_token_id=token_id)
                result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                translated_text = "".join(result)
                print(translated_text)
                result_element["text"] = translated_text
            except:
                result_element["text"] = text_to_translate
        result_to_report["result"]["texts"].append(result_element)
    end_time = datetime.datetime.now()
    result_to_report["result"]["device"] = DEVICE
    result_to_report["result"]["duration"] = (end_time - start_time).total_seconds()
    result_to_report["result"]["repository"] = REPOSITORY
    result_to_report["result"]["version"] = VERSION
    result_to_report["result"]["library"] = LIBRARY
    result_to_report["result"]["model"] = MODEL
    result_to_report["result"]["apiversion"] = APIVERSION
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

# Compare transformers and argos

texttotranslate = "The siege of Guînes took place from May to July 1352, when a French army under Geoffrey de Charny unsuccessfully attempted to recapture the French castle at Guînes which had been seized by the English the previous January. The siege was part of the Hundred Years' War and took place during the uneasy and oft-broken truce of Calais.  The English had taken the strongly fortified castle during a period of nominal truce, and the English king, Edward III, decided to keep it. Charny led 4,500 men and retook the town, but could not blockade the castle. After two months of fierce fighting, a large English night attack on the French camp inflicted a heavy defeat and the French withdrew. Guînes was incorporated into the Pale of Calais. The castle was besieged by the French in 1436 and 1514 but was relieved each time, before falling to the French in 1558."
from_code = "en"
to_code = "de"

import os
import time
ct = time.time()
def timediff():
    global ct
    now = time.time()
    diff = now - ct
    ct = now
    return str(diff) + "\n"

# For using GPU https://github.com/huggingface/transformers/issues/2704
print("LOADING TRANSFORMERS ...")
print(timediff(), "Importing Torch ...")
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(timediff(), "Device:", device)
print(timediff(), "Loading transformers ...")
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
print(timediff(), "Loading model ...")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B").to(device)
print(timediff(), "Loading tokenizer ...")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")

print(timediff(), "READY")

# Transformers
print("Translating with Transformers ...")
#print("Encoding " + from_code + " ...")
#tokenizer.src_lang = from_code
#encoded = tokenizer(texttotranslate, return_tensors="pt").to(device)
#print("Creating tokens for " + to_code + " ...")
#generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(to_code))
#print("Decoding tokens ...")
#result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
#print(timediff(), result)

sentences = [x for x in texttotranslate.split('.')]
print("Looping over sentences of long text (", len(sentences), " sentences) ...")
tokenizer.src_lang = from_code
for sentence in sentences:
    print("Encoding en ...")
    encoded = tokenizer(sentence, return_tensors="pt").to(device)
    print("Creating tokens for " + to_code + " ...")
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(to_code))
    print("Decoding tokens ...")
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(result)

print(timediff(), "DONE.")


# Transformers (Genauer, schluckt längere Texte)
# 3.4070842266082764 Die Belagerung von Guînes fand von Mai bis Juli 1352 statt, als eine französische Armee unter Geoffrey de Charny erfolglos versuchte, das französische Schloss bei Guînes zurückzuerobern, das im Januar zuvor von den Engländern eingenommen worden war.

# Argos (Schneller, kann mit mehr Text umgehen)
# 2.2560300827026367 Die Belagerung von Guînes fand von Mai bis Juli 1352 statt, als eine französische Armee unter Geoffrey de Charny erfolglos versuchte, die französische Burg in Guînes zu erobern, die vom Englischen im vergangenen Januar beschlagnahmt worden war.
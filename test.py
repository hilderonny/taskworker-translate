import time

ct = time.time()

def timediff():
    global ct
    now = time.time()
    diff = now - ct
    ct = now
    return str(diff) + "\n"

# For using GPU https://github.com/huggingface/transformers/issues/2704
print(timediff(), "Importing Torch ...")
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(timediff(), "Device:", device)

print(timediff(), "Loading transformers ...")
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

hi_text = "जीवन एक चॉकलेट बॉक्स की तरह है।"
chinese_text = "生活就像一盒巧克力。"

print(timediff(), "Loading model ...")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B").to(device)
print(timediff(), "Loading tokenizer ...")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")

# translate Hindi to French
tokenizer.src_lang = "hi"
print(timediff(), "Encoding hi ...")
encoded_hi = tokenizer(hi_text, return_tensors="pt").to(device)
print(timediff(), "Creating tokens for fr ...")
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("fr"))
print(timediff(), "Decoding tokens ...")
result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(timediff(), result)
print(timediff(), "Creating tokens for de ...")
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("de"))
print(timediff(), "Decoding tokens ...")
result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(timediff(), result)
# => "La vie est comme une boîte de chocolat."

# translate Chinese to English
tokenizer.src_lang = "zh"
print(timediff(), "Encoding zh ...")
encoded_zh = tokenizer(chinese_text, return_tensors="pt").to(device)
print(timediff(), "Creating tokens for en ...")
generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
print(timediff(), "Decoding tokens ...")
result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(timediff(), result)
# => "Life is like a box of chocolate."

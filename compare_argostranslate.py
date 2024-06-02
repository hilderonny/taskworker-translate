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

print("LOADING ARGOS ...")
os.environ['ARGOS_DEVICE_TYPE'] = "cuda"
print(timediff(), "Importing dependencies ...")
import argostranslate.package
import argostranslate.translate
# Download and install Argos Translate package
print(timediff(), "Updating packages ...")
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    filter(
        lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
    )
)
argostranslate.package.install_from_path(package_to_install.download())

print(timediff(), "READY")

# Argos
print("Translating with Argos ...")
#translatedText = argostranslate.translate.translate(texttotranslate, from_code, to_code)
#print(timediff(), translatedText)

sentences = [x for x in texttotranslate.split('.')]
print("Looping over sentences of long text (", len(sentences), " sentences) ...")
for sentence in sentences:
    translatedText = argostranslate.translate.translate(sentence, from_code, to_code)
    print(translatedText)

print(timediff(), "DONE.")

# 2.2560300827026367 Die Belagerung von Guînes fand von Mai bis Juli 1352 statt, als eine französische Armee unter Geoffrey de Charny erfolglos versuchte, die französische Burg in Guînes zu erobern, die vom Englischen im vergangenen Januar beschlagnahmt worden war.
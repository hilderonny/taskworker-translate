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

# Long text
char_count = 1000
very_long_text = "The siege of Guînes took place from May to July 1352, when a French army under Geoffrey de Charny unsuccessfully attempted to recapture the French castle at Guînes which had been seized by the English the previous January. The siege was part of the Hundred Years' War and took place during the uneasy and oft-broken truce of Calais.  The English had taken the strongly fortified castle during a period of nominal truce, and the English king, Edward III, decided to keep it. Charny led 4,500 men and retook the town, but could not blockade the castle. After two months of fierce fighting, a large English night attack on the French camp inflicted a heavy defeat and the French withdrew. Guînes was incorporated into the Pale of Calais. The castle was besieged by the French in 1436 and 1514 but was relieved each time, before falling to the French in 1558.  Background Since the Norman Conquest of 1066, English monarchs had held titles and lands within France, the possession of which made them vassals of the kings of France.[1] Following a series of disagreements between Philip VI of France (r. 1328–1350) and Edward III of England (r. 1327–1377), on 24 May 1337 Philip's Great Council in Paris agreed that the lands held by Edward in France should be taken back into Philip's hands because Edward was in breach of his obligations as a vassal. This marked the start of the Hundred Years' War, which was to last 116 years.[2][3][4] After nine years of inconclusive but expensive warfare, Edward landed with an army in northern Normandy in July 1346.[5] He then undertook the Crécy campaign, to the gates of Paris and north across France.[6][7] The English turned to fight Philip's much larger army at the Battle of Crécy, where the French were defeated with heavy loss.[8]  Edward needed a port where his army could regroup and be resupplied from the sea. The Channel port of Calais suited this purpose; it was also highly defensible and would provide a secure entrepôt into France for English armies. Calais could be easily resupplied by sea and defended by land.[9][10] Edward's army laid siege to the port in September 1346. With French finances and morale at a low ebb after Crécy, Philip failed to relieve the town, and the starving defenders surrendered on 3 August 1347.[11][12] By 28 September the Truce of Calais, intended to bring a temporary halt to the fighting, had been agreed.[13] It was to run for nine months to 7 July 1348 but was extended repeatedly.[14] The truce did not stop ongoing naval clashes between the two countries, nor small-scale fighting in Gascony and Brittany.[15][16]  In July 1348, a member of the King's Council, Geoffrey de Charny, was put in charge of all French forces in the northeast.[17] Despite the truce being in effect Charny hatched a plan to retake Calais by deception and bribed Amerigo of Pavia, an Italian officer of the city garrison, to open a gate for a force led by him.[18][19][20] The English king became aware of the plot, crossed the Channel and led his household knights and the Calais garrison in a surprise counter-attack.[21][22] When the French approached on New Year's Day 1350 they were routed by this smaller force, with significant losses and all their leaders captured or killed; Charny was among the captured.[23]  In late 1350 Raoul, Count of Eu, the Grand Constable of France, returned after more than four years in English captivity. He was personally on parole from Edward, pending his ransom's handover. This was a tremendous amount, rumored to have been 80,000 écus; more than Raoul could afford. It had been agreed that he would instead hand over the town of Guînes, 6 miles (9.7 km) from Calais, which was in his possession. This was a common method of settling ransoms. Guînes had an extremely strong keep and was the leading fortification in the French defensive ring around Calais. English possession would go a long way to securing Calais against further surprise assaults. Guînes was of little financial value to Raoul, and it was clear that Edward was prepared to accept it instead of a full ransom payment only because of its strategic position.[24][25] Angered by the attempt to weaken the blockade of Calais, the new French king, John II, had Raoul executed for treason, preventing the transaction from taking place. This interference by the crown in a nobleman's personal affairs, especially one of such high status, caused uproar in France.[26]  English attack A circular Medieval stone tower with a clock near the top The keep at Guînes in 2007 In early January 1352 a band of freelancing English soldiers, led by John of Doncaster, seized the town of Guînes by midnight escalade. The fortifications at Guînes were often used as quarters for English prisoners. According to some contemporary accounts Doncaster had been employed as forced labor there after being taken captive earlier in the war and had used the opportunity to examine the town's defenses. After gaining his freedom he had remained in France as a member of the garrison of Calais, as he had been exiled from England for violent crimes.[27][28] One of these sources suggests that Doncaster learned the details of Guînes' defenses through an affair with a French washerwoman.[29] The French garrison of Guînes was not expecting an attack and Doncaster's party crossed the moat, scaled the walls, killed the sentries, stormed the keep, released the English prisoners there, and took over the whole castle.[27]  The French were furious: the acting commander, Hugues de Belconroy, was drawn and quartered for dereliction of duty, at the behest of Charny, who had returned to France after being ransomed from English captivity. French envoys rushed to London to deliver a strong protest to Edward on 15 January.[30][31] Edward was thereby put in a difficult position. The English had been strengthening the defenses of Calais with the construction of fortified towers or bastions at bottlenecks on the roads through the marshes to the town.[32] These could not compete with the strength of the defenses at Guînes, which would greatly improve the security of the English enclave around Calais. However, retaining it would be a flagrant breach of the truce then in force. Edward would suffer a loss of honor and possibly a resumption of open warfare, for which he was unprepared. He therefore ordered the English occupants to hand Guînes back.[27]  By coincidence, the English Parliament was scheduled to meet, with its opening session on 17 January. Several members of the King's Council made fiery, warmongering speeches and the parliament was persuaded to approve three years of war taxes. Reassured that he had adequate financial backing, Edward changed his mind. By the end of January, the Captain of Calais had fresh orders: to take over the garrisoning of Guînes in the King's name. Doncaster was pardoned and rewarded. Determined to strike back, the French took desperate measures to raise money and set about raising an army.[31]  French attack  The motte and keep of Guînes castle in 2012 The outbreak of hostilities at Guînes caused fighting to also flare up in Brittany and the Saintonge area of south-west France, but the main French effort was against Guînes. Geoffrey de Charny was again put in charge of all French forces in the north-east. He assembled an army of 4,500 men, including 1,500 men-at-arms and a large number of Italian crossbowmen. By May the 115 men of the English garrison, commanded by Thomas Hogshaw, were under siege. The French reoccupied the town, but found it difficult to approach the castle. The marshy ground and many small waterways made it difficult to approach from most directions, while facilitating waterborne supply and reinforcement for the garrison. Charny decided that the only practicable approach was via the main entrance facing the town, which was defended by a strong barbican. He had a convent a short distance away converted into a fortress, surrounded by a stout palisade, and positioned catapults and cannons there.[33]  By the end of May the English authorities, concerned by these preparations, raised a force of more than 6,000 which was gradually shipped to Calais. From there they harassed the French in what the modern historian Jonathan Sumption describes as 'savage and continual fighting' throughout June and early July. In mid-July a large contingent of troops arrived from England, and, reinforced by much of the Calais garrison, they were successful in approaching Guînes undetected and launching a night attack on the French camp. Many Frenchmen were killed and a large part of the palisade around the convent was destroyed. Shortly afterwards Charny abandoned the siege, leaving a garrison to hold the convent.[34]  The French captured and slighted a newly built English tower at Fretun, 3 miles (4.8 km) south-west of Calais, then retreated to Saint-Omer, where their army disbanded.[34] During the rest of the year the English expanded their enclave around Calais, building and strengthening fortifications on all the access routes through the marshes around Calais, forming what became the Pale of Calais. The potential offensive threat posed by Calais caused the French to garrison 60 fortified positions in an arc around the town, at ruinous expense.[35]  Aftermath The war also went badly for the French on other fronts and, encouraged by the new pope, Innocent VI, a peace treaty was negotiated at Guînes beginning in early 1353. On 6 April 1354 a draft was agreed. This Treaty of Guînes would have ended the war, very much in the favour of England. French and English ambassadors travelled to Avignon that winter to ratify the treaty in the presence of the Pope. This did not occur as the French king was persuaded that another round of warfare might leave him in a better negotiating position and withdrew his representatives.[36]  Charny was killed in 1356 at the Battle of Poitiers, when the French royal army was defeated by a smaller Anglo-Gascon force commanded by Edward's son, the Black Prince, and John was captured.[37] In 1360, the Treaty of Brétigny ended the war, with vast areas of France being ceded to England; including Guînes and its county which became part of the Pale of Calais.[38] The castle was besieged by the French in 1436 and 1514, but was relieved each time.[39] Guînes remained in English hands until it was recaptured by the French in 1558.[11]"
long_text = very_long_text[:char_count]
print(timediff(), "Translating long text (", char_count, " chars) ...")
tokenizer.src_lang = "en"
print(timediff(), "Encoding en ...")
encoded_en = tokenizer(long_text, return_tensors="pt").to(device)
print(timediff(), "Creating tokens for de ...")
generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id("de"))
print(timediff(), "Decoding tokens ...")
result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(timediff(), result)

# Loop
sentences = [x for x in very_long_text.split('.')]
print(timediff(), "Looping over sentences of long text (", len(sentences), " sentences) ...")
tokenizer.src_lang = "en"
for sentence in sentences:
    print(timediff(), "Encoding en ...")
    encoded_en = tokenizer(sentence, return_tensors="pt").to(device)
    print(timediff(), "Creating tokens for de ...")
    generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id("de"))
    print(timediff(), "Decoding tokens ...")
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(timediff(), result)

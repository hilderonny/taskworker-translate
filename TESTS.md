# Comparison of Transformers vs. Argos Translate

Source text:

```
The siege of Guînes took place from May to July 1352, when a French army under Geoffrey de Charny unsuccessfully attempted to recapture the French castle at Guînes which had been seized by the English the previous January.

The siege was part of the Hundred Years' War and took place during the uneasy and oft-broken truce of Calais.

The English had taken the strongly fortified castle during a period of nominal truce, and the English king, Edward III, decided to keep it.

Charny led 4,500 men and retook the town, but could not blockade the castle.

After two months of fierce fighting, a large English night attack on the French camp inflicted a heavy defeat and the French withdrew.

Guînes was incorporated into the Pale of Calais.

The castle was besieged by the French in 1436 and 1514 but was relieved each time, before falling to the French in 1558.
```

The text was split into sentences because the transformer model could not handle large texts. Argos split the text into paragraphs internally.

Transformers took 11 seconds in sum, Argos Translate took 4 seconds.

Transformers results were more exactly than the Argos ones. Transformers sometimes contains punctuation marks. Argos needs Python 3.11 and additional CUDA downloads to work, transformers has all dependencies within itself and works with python 3.12

Transformers result:

```
Die Belagerung von Guînes fand von Mai bis Juli 1352 statt, als eine französische Armee unter Geoffrey de Charny erfolglos versuchte, das französische Schloss bei Guînes zurückzuerobern, das im Januar von den Engländern beschlagnahmt worden war.

Die Belagerung war Teil des Hundertjährigen Krieges und fand während der unruhigen und oft gebrochenen Waffenruhe von Calais statt.

Die Engländer hatten das stark befestigte Schloss während einer Periode des nominalen Waffenstillstands übernommen, und der englische König Edward III. beschloss, es zu behalten.

Charny führte 4.500 Männer und restaurierte die Stadt, konnte aber das Schloss nicht blockieren

Nach zwei Monaten heftigen Kämpfen verursachte ein großer englischer Nachtangriff auf das französische Lager eine schwere Niederlage und die Franzosen zogen sich zurück.

Guînes wurde in den Palast von Calais integriert

Das Schloss wurde 1436 und 1514 von den Franzosen belagert, wurde aber jedes Mal gelindert, bevor es 1558 an die Franzosen fiel.

ist
```

Argos translation:

```
Die Belagerung von Guînes fand von Mai bis Juli 1352 statt, als eine französische Armee unter Geoffrey de Charny erfolglos versuchte, die französische Burg in Guînes zu erobern, die vom Englischen im vergangenen Januar beschlagnahmt worden war

Die Belagerung war Teil des Hundertjährigen Krieges und fand während der uneasy und der zerbrochenen Truce von Calais statt

Das Englische hatte das stark befestigte Schloss während einer Periode der nominalen Truce genommen, und der englische König, Edward III, beschlossen, es zu halten

Charny führte 4.500 Männer und erholte die Stadt, aber konnte nicht blockieren das Schloss

Nach zwei Monaten heftigen Kämpfen, ein großer englischer Nachtangriff auf das französische Lager hatte eine schwere Niederlage und die Franzosen zogen zurück

Guînes wurde in den Pale von Calais integriert

Die Burg wurde 1436 und 1514 von den Franzosen belagert, wurde aber jedes Mal entlastet, bevor sie 1558 in die Franzosen fiel
```

Transformers seems to devliver the better results.

# Translating texts with multiple languages

The following text was given

```
Englisch:
The morning breeze was gentle, carrying the scent of blooming flowers. Birds chirped merrily, welcoming the new day. The streets were slowly filling with people starting their daily routines.

Mandarin (简体中文):
清晨的微风温柔，带着盛开的花香。鸟儿们欢快地鸣叫，迎接新的一天。街道上逐渐热闹起来，人们开始了他们的日常生活。

Hindi (हिन्दी):
सुबह की हवा हल्की थी, जिसमें फूलों की महक बसी थी। पक्षी खुशी-खुशी चहचहा रहे थे, नए दिन का स्वागत कर रहे थे। सड़कें धीरे-धीरे लोगों से भरने लगीं, जो अपनी दिनचर्या शुरू कर रहे थे।

Spanisch:
La brisa de la mañana era suave, llevando el aroma de las flores en flor. Los pájaros cantaban alegremente, dando la bienvenida al nuevo día. Las calles comenzaban a llenarse de gente que iniciaba sus rutinas diarias.

Arabisch (العربية):
كانت نسيم الصباح لطيفًا، يحمل رائحة الأزهار المتفتحة. كانت الطيور تزقزق بسعادة، ترحب باليوم الجديد. كانت الشوارع تزدحم ببطء بالناس الذين يبدأون روتينهم اليومي.

Französisch:
La brise matinale était douce, emportant avec elle le parfum des fleurs épanouies. Les oiseaux chantaient joyeusement, accueillant le nouveau jour. Les rues se remplissaient lentement de gens commençant leurs activités quotidiennes.

Russisch (Русский):
Утренний ветерок был нежным, неся аромат цветущих цветов. Птицы весело щебетали, приветствуя новый день. Улицы медленно наполнялись людьми, начинавшими свои повседневные дела.

Portugiesisch:
A brisa da manhã era suave, carregando o perfume das flores em flor. Os pássaros cantavam alegremente, dando as boas-vindas ao novo dia. As ruas estavam lentamente se enchendo de pessoas iniciando suas rotinas diárias.

Bengalisch (বাংলা):
সকালের হাওয়া মৃদু ছিল, যা ফুলের সুবাস নিয়ে আসছিল। পাখিরা আনন্দের সাথে ডাকছিল, নতুন দিনকে স্বাগত জানাচ্ছিল। ধীরে ধীরে রাস্তাগুলো মানুষে ভরে উঠছিল, যারা তাদের দৈনন্দিন কাজ শুরু করছিল।

Urdu (اردو):
صبح کی ہوا ہلکی تھی، جو کھلتے ہوئے پھولوں کی خوشبو لے کر آئی تھی۔ پرندے خوشی سے چہچہا رہے تھے، نئے دن کا استقبال کر رہے تھے۔ سڑکیں آہستہ آہستہ لوگوں سے بھرنے لگیں جو اپنی روزمرہ کی سرگرمیاں شروع کر رہے تھے۔

Türkisch:
Sabah esintisi hafifti, çiçeklerin kokusunu taşıyordu. Kuşlar neşeyle ötüşerek yeni günü karşılıyorlardı. Sokaklar yavaş yavaş günlük işlerine başlayan insanlarla dolmaya başladı.

Polnisch:
Poranny wiatr był delikatny, niosąc ze sobą zapach kwitnących kwiatów. Ptaki wesoło ćwierkały, witając nowy dzień. Ulice powoli wypełniały się ludźmi rozpoczynającymi swoje codzienne obowiązki.

Griechisch (Ελληνικά):
Το πρωινό αεράκι ήταν απαλό, μεταφέροντας το άρωμα των ανθισμένων λουλουδιών. Τα πουλιά κελαηδούσαν χαρούμενα, καλωσορίζοντας τη νέα μέρα. Οι δρόμοι σιγά-σιγά γέμιζαν με ανθρώπους που ξεκινούσαν τις καθημερινές τους δραστηριότητες.

Japanisch (日本語):
朝のそよ風は穏やかで、咲き誇る花の香りを運んでいました。鳥たちは楽しげにさえずり、新しい一日を迎えていました。通りは、日常の活動を始める人々でゆっくりと賑わい始めました。

Latein (Latine):
Aurorae aura lenis erat, flores recentes odores ferens. Aves laetitia cantabant, novum diem salutantes. Viae paulatim hominibus replentur, cotidianos labores incipientibus.
```

The result was

```
auf Englisch:
Der morgendliche Wind war sanft, trug den Duft von blühenden Blumen.Vögel zitterten freudig, begrüßten den neuen Tag.Die Straßen füllten sich langsam mit Menschen, die ihre täglichen Routinen begannen.

Mandarinisch (简体中文):
Die frühen Morgenwinde sind sanft, mit einem blühenden Duft. Die Vögel klingeln fröhlich und begrüßen den neuen Tag. Die Straßen werden allmählich belebt und die Menschen beginnen ihren Alltag.

In Hindi (Englisch)
Der Wind des Morgens war hell, mit einer Fülle von Blumen. Die Vögel waren glücklich und begrüßten den neuen Tag. Die Straßen begannen langsam mit Menschen zu füllen, die ihre Routine begannen.

auf Spanisch:
Der Wind des Morgens war sanft, mit dem Duft der blühenden Blumen.Die Vögel sangen freudig und begrüßten den neuen Tag.Die Straßen begannen, sich mit Menschen zu füllen, die ihre täglichen Routinen begannen.

Arabisch (arabisch)
Es war ein schöner Morgen, der den Duft der blühenden Blumen trägt. Die Vögel waren glücklich, begrüßten den neuen Tag. Die Straßen wurden langsam von Menschen gefüllt, die ihre tägliche Routine begannen.

auf Französisch:
Die Morgenbrise war sanft und trug den Duft der blühenden Blumen mit sich. Die Vögel sangen fröhlich und begrüßten den neuen Tag. Die Straßen füllten sich langsam mit Menschen, die ihre täglichen Aktivitäten begannen.

Russisch (Russisch)
Der morgendliche Wind war sanft, mit dem Duft der blühenden Blumen. Die Vögel freuten sich und begrüßten den neuen Tag. Die Straßen füllten sich langsam mit Menschen, die ihre täglichen Angelegenheiten begannen.

auf Portugiesisch:
Der Wind des Morgens war sanft und trug den Duft der blühenden Blumen. Die Vögel sangen fröhlich und begrüßten den neuen Tag. Die Straßen füllten sich langsam mit Menschen, die ihre täglichen Routinen begannen.

Bengalische Sprache (Bengali)
Die Morgenluft war weich, was den Duft der Blumen mit sich brachte. Die Vögel riefen mit Freude, begrüßten den neuen Tag. Langsam wurden die Straßen von Menschen gefüllt, die ihre tägliche Arbeit begannen.

Der Urdu (Arduo)
Die Morgenluft war hell, die mit dem Duft der Blumen kam, die Vögel zitterten mit Freude, begrüßten den neuen Tag, die Straßen wurden langsam von Menschen gefüllt, die ihre täglichen Aktivitäten begannen.

auf Türkisch:
Die Morgendämmerung war leicht, sie trug den Duft der Blumen. Die Vögel begrüßten den neuen Tag mit Freude. Die Straßen begannen langsam mit Menschen zu füllen, die ihre täglichen Aufgaben begannen.

Polnisch zu:
Der Morgenwind war mild und trug den Duft der blühenden Blumen mit sich. Die Vögel klatschten fröhlich und begrüßten den neuen Tag. Die Straßen füllten sich langsam mit Menschen, die ihre täglichen Pflichten begannen.

Griechisch (auf Griechisch)
Der morgendliche Wind war sanft, trug den Duft der blühenden Blumen. Die Vögel klingelten fröhlich und begrüßten den neuen Tag. Die Straßen füllten sich langsam mit Menschen, die ihre täglichen Aktivitäten begannen.

auf Japanisch (Japanese):
Der Wind des Morgens war ruhig und trug den Duft von blühenden Blumen. Die Vögel begrüßten den neuen Tag mit Freude. Die Straßen begannen langsam und voller Menschen, die ihre täglichen Aktivitäten begannen.

Latein (auf Latein)
Aurorae aura lenis erat, floras recentes odores ferens. Aves laetitia cantabant, novum diem salutantes. Viae paulatim hominibus replentur, cotidianos labores incipientibus.
```

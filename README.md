# taskworker-translate
Worker for taskbridge which can handle "translate" tasks

See:

1. https://huggingface.co/facebook/m2m100_1.2B
2. https://github.com/ymoslem/DesktopTranslator?tab=readme-ov-file#m2m-100-multilingual-model
3. https://github.com/argosopentech/argos-translate


## Source code setup

First install Python 3.12.

```
python -m venv python-venv
python-venv/Scripts/activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.44.2 sentencepiece==0.2.0
```

The last commands depend on the operating system, see https://pytorch.org/get-started/locally/ and can download several gigabytes.

## Running

Running the program the first time, ai models with about 5 GB must be downloaded

```sh
python translate.py --apiurl http://192.168.178.39:42000/api/ --sourcelanguage en --targetlanguage de
```

## Task format

 ```js
 task = {
    id: "36b8f84d-df4e-4d49-b662-bcde71a8764f",
    data: {
        texts: [
            "Hello world!",
            "Here I am."
        ]
    },
    result: {
        device: "cuda:0",
        duration: 12,
        repository: "https://github.com/hilderonny/taskworker-translate",
        version: "1.1.0",
        library: "transformers",
        model: "facebook/m2m100_1.2B",
        texts: [
            "Hallo Welt!",
            "Hier bin ich."
        ]
    }
 }
 ```

|Property|Description|
|---|---|
|`data.texts`|Array of texts to translate. Each element should be a separate sentence and should be no longer than **200** characters|
|`result.device`|Device type which processed the translation. Can be `cuda:0` for GPU processing on the first NVidia graphic card or `cpu` for normal CPU processing|
|`result.repository`|Repository URL of the worker which processed the task|
|`result.version`|Version of the worker program used for processing|
|`result.library`|NPM library used internally for AI processing|
|`result.model`|AI model used for processing|
|`result.texts`|List of translated texts. The list is of the same size as `data.texts` and the elements are in the same order so that there is a direct correlation between the input and output arrays|


## Comparison of Transformers vs. Argos Translate

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
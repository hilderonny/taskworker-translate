# taskworker-translate

Worker for taskbridge which can handle tasks of type `translate`.

## Task format

```json
{
    ...
    "type" : "translate",
    "worker" : "ROG",
    "data" : {
        "sourcelanguage" : "en",
        "targetlanguage" : "de",
        "texts" : [
            "Line 1",
            "",
            "Ligne 3"
        ]
    },
    ...
    "result" : {
        "texts" : [
            {
                "text" : "Zeile 1",
                "sourcelanguage" : "en"
            },
            {
                "text" : ""
            },
            {
                "text" : "Zeile 3",
                "sourcelanguage" : "fr"
            }
        ],
        "device" : "cuda:0",
        "duration" : 12,
        "repository" : "https://github.com/hilderonny/taskworker-translate",
        "version" : "1.2.0",
        "library" : "transformers-4.44.2",
        "model" : "facebook/m2m100_1.2B",
        "apiversion" : "v1"
    }
}
```

The `type` must be `translate`.

`worker` contains the unique name of the worker.

The worker expects a `data` object which consists of the `targetlanguage` in which all texts are to be translated into and an array of `texts` to be translated.
The `sourcelanguage` is optional and forces the worker to translate from this language without detecting it. When this property is missing, for each line the language gets detected.
The `targetlanguage` needs to be a two digit ISO code.
The `texts` array should consist of sentences or short paragraphs. An element can also be empty.

When the worker finishes the task, it sends back a `result` property. This property is an object. It contains an array `texts` which is of the same size as the `data.texts` property above. For each element in the data array there is an equivalent element in the results array. The arrays are ordered the same way. Each element is an object containing the translated `text` and the detected `sourcelanguage`of the text snippet expressed as zwo digits ISO code. Empty lines in the data array will be transferred into the result array without any language information. In `apiversion` there is the used version of the Task Bridge API.

## Installation

First install Python 3.12. The run the following commands in the folder of the downloaded repository.

```sh
python3.12 -m venv python-venv
python-venv\Scripts\activate # Windows
source ./python-venv/bin/activate # Linux
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.44.2 sentencepiece==0.2.0 langdetect==1.0.9
```

The `pip` commands depend on the operating system, see https://pytorch.org/get-started/locally/ and can download several gigabytes.

## Running

Running the program the first time, ai models with about 5 GB gets downloaded automatically.

```sh
python translate.py --taskbridgeurl http://192.168.178.39:42000/ --worker ROG --device cuda:0
```

The `device` defines which device to use for processing. Can be `cpu`, `cuda` or `cuda:X` where `X` is the index of the graphics card to use.

## Literature

1. https://huggingface.co/facebook/m2m100_1.2B
2. https://github.com/ymoslem/DesktopTranslator?tab=readme-ov-file#m2m-100-multilingual-model
3. https://github.com/argosopentech/argos-translate


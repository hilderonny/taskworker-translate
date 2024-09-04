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
pip install transformers==4.44.2 sentencepiece==0.2.0 langdetect==1.0.9
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

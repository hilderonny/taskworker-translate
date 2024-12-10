# taskworker-translate

Worker for taskbridge which can handle tasks of type `translate`.

## Result format

When calling the TaskBridge `/api/tasks/complete/:id` API, the following JSON structure is sent to the endpoint.

```json
{
  "result" : {
    "texts" : [
      {
        "sourcelanguage" : "en",
        "text" : "Text der ersten Zeile auf Englisch"
      },
      {
        "sourcelanguage" : "ru",
        "text" : "Text der zweiten Zeile auf Russisch"
      },
      {
        "sourcelanguage" : "ar",
        "text" : "Text der dritten Zeile auf Arabisch"
      }
    ],
    "device" : "cuda",
    "duration" : 1.6,
    "repository" : "https://github.com/hilderonny/taskworker-translate",
    "version" : "1.3.0",
    "library": "transformers-4.44.2",
    "model": "facebook/m2m100_1.2B"
  }
}
```

|Property|Description|
|---|---|
|`texts`|Array of translated texts. Same size as request array|
|`texts.sourcelanguage`|Detected language of the source text as 2 digit ISO code. Only set, when source language was not forced via request|
|`texts.text`|Text translated into target language|
|`device`|`cuda` for GPU processing and `cpu` for CPU processing|
|`duration`|Time in seconds for the processing|
|`repository`|Source code repository of the worker|
|`version`|Version of the worker|
|`library`|Library used to perform translation|
|`model`|AI model used for translation|

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

Adopt the shell script `translate.sh` to your needs and create SystemD config files (if you want tu run the worker as Linux service).

**/etc/systemd/system/taskworker-translate.service**:

```
[Unit]
Description=Forensic Task Worker - Text Translator

[Service]
ExecStart=/taskworker-translate/translate.sh
Restart=always
User=user
WorkingDirectory=/taskworker-translate/

[Install]
WantedBy=multi-user.target
```

Finally register and start the services.

```
chmod +x ./translate.sh
sudo systemctl daemon-reload
sudo systemctl enable taskworker-translate.service
sudo systemctl start taskworker-translate.service
```

## Running

Running the program the first time, ai models with about 5 GB gets downloaded automatically.

```sh
python translate.py --taskbridgeurl http://192.168.178.39:42000/ --worker ROG --device cuda:0
```

The `device` defines which device to use for processing. Can be `cpu`, `cuda` or `cuda:X` where `X` is the index of the graphics card to use.

## Docker

There is also a [Docker image](https://hub.docker.com/repository/docker/hilderonny2024/taskworker-translate) you can use. It is tested with Docker Desktop on Windows using WSL2.

```sh
docker run --gpus all -e taskbridgeurl=http://192.168.178.39:42000/ -e worker=ROG hilderonny2024/taskworker-translate
```

The first time you run the application the AI model needs to be downloaded (about 5 GB), which can take some time. You can watch the progress by opening a terminal in the docker container and running

```sh
ls -la /root/.cache/huggingface/hub/models--facebook--m2m100_1.2B/blobs/
```

## Literature

1. https://huggingface.co/facebook/m2m100_1.2B
2. https://github.com/ymoslem/DesktopTranslator?tab=readme-ov-file#m2m-100-multilingual-model
3. https://github.com/argosopentech/argos-translate


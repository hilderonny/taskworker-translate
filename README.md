# taskworker_translate
Worker for taskbridge which can handle "translate" tasks

See:

1. https://huggingface.co/facebook/m2m100_1.2B
2. https://github.com/ymoslem/DesktopTranslator?tab=readme-ov-file#m2m-100-multilingual-model
3. https://github.com/argosopentech/argos-translate


## Source code setup

```
python -m venv python-venv
python-venv/Scripts/activate
pip install -r requirements.txt
pip install sentencepiece transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

The last command depends on the operating system, see https://pytorch.org/get-started/locally/ and can download several gigabytes.

Running the program the first time, ai models with about 5 GB must be downloaded
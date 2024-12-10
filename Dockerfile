FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
RUN apt update && apt install -y python3 python3-pip
RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir transformers sentencepiece langdetect

WORKDIR /app

ENV taskbridgeurl=http://127.0.0.1:42000/
ENV worker=DOCKER
ENV device=cuda:0

COPY translate.py /app

CMD [ "sh" , "-c", "python3 -u translate.py --taskbridgeurl ${taskbridgeurl} --worker ${worker} --device ${device}" ]

# Building
# docker build -t hilderonny2024/taskworker-translate .

# First time run to enable GPUs
# docker run --gpus all -e taskbridgeurl=http://127.0.0.1:42000/ -e worker=RH-WORKBOOK-DOCKER hilderonny2024/taskworker-translate

# Publishing
# docker login
# docker push hilderonny2024/taskworker-translate
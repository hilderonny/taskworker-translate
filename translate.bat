@ECHO OFF
SET HF_HOME=.\models
.\python\python.exe .\translate.py --taskbridgeurl http://127.0.0.1:42000/ --worker LOCALHOST --device cuda:0
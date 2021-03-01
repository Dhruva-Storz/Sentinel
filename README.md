<h1>Sentinel: Data mining system for Twitter-based event summarisation and sentiment analysis</h1>
  
This repo documents our group project in the software engineering course at the MSc Artificial Intelligence, Imperial College London.

<h2>How to run webapp</h2>

<h3> Installing Requirements </h3>

Install requirements via pip from the root folder and spacy module

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

<h3> Running the Webapp </h3>

The webapp mockup can be run by first navigating into the sentinel folder, and then running

```
python3 app.py
```

The localhost port can be specified using 

```
python3 mockup.py -p PORT
```

With the default port being 5000.
After the flask server initializes, the webapp can be accessed by going to http://localhost:$PORT$/ where $PORT$ is 5000 if the port is not specified, and the specified port if it is. 


NOTE: Repo is not currently functional due to missing models and API credentials

Todo: 
Add gdown links to SA model
Add gdown links to summarization pretrained model
Add means of specifying googleAPI and twitter API keys from argparse or config file.


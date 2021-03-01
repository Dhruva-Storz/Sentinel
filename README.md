<h1>Sentinel: Data mining system for Twitter-based event summarisation and sentiment analysis</h1>
  
This repo documents our group project in the software engineering course at the MSc Artificial Intelligence, Imperial College London.

<h2>How to run webapp</h2>

<h3> Installing Requirements </h3>

Install requirements via pip from the root folder and spacy module

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

<h3> Obtaining API keys </h3>

To run the webapp, you need a twitter API key (necessary to scrape twitter) and a Google custom search API key (optional for the pretty pictures).

I would share my own but github key scrapers make this impossible.
They can be obtained for free from twitter and google developer pages respectively.
Once they are obtained, fill them into the credentials.json file in the Sentinel folder. 

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


NOTE: Repo is not currently functional due to missing models

Todo: 
Add gdown links to SA model
Add gdown links to summarization pretrained model
Proper accreditation of sources, ideas and papers


<h1>Sentinel: Data mining system for Twitter-based event summarisation and sentiment analysis</h1>
  
This repo documents our group project in the software engineering course at the MSc Artificial Intelligence, Imperial College London.

You can see the webapp by following the link below

https://youtu.be/CD04QrXFRpk

Note: link doesnt show summarization and only shows results for 20 tweets as the inference at the time of recording this video was done on a cpu. A new video will be uploaded with the full functionality soon. 

<h2> How it works </h2>

The webapp takes a query word or sentence and scrapes twitter for tweets using twitter's API. Specific scraping parameters can be selected under advanced. Once a large number of tweets are scraped, we run our custom sentiment analysis, topic analysis and summarization AI models on the scraped text. The AI models compute overall sentiment distributions, sentiments over time, prominent and common topics (LDA) and a summary of the tweets of obtained. 

Due to the time it takes for our AI systems to thoroughly analyse your query, a loading screen displays sample tweets as you wait with an image related to the query in the background. The image is unique to each query and is obtained from Google's custom search API by searching for the first high resolution public access image (to ensure we are allowed to display it on the webapp) and displaying it on the loading screen. 

<h3> Frontend Design </h3>

The frontend is built on a flask backend with Jinja. The frontend itself is built with HTML, CSS and javascript with d3.js for visualizing the AI's outputs. 

<h3> Backend Design </h3>

Due to needing to run several AI models to perform analysis, we run them in paralell to cut down processing time. All communication between the python backend and the HTML frontend is done asynchronously as well to facilitate loading screens and dynamic updating of the results (the summarization usually takes longer than the other two). Asynchronous communication between the frontend and backend is done using flask-socketio (https://github.com/miguelgrinberg/Flask-SocketIO) and custom javascript code. 

Twitter scraping was done using tweepy. 

<h3> Sentiment Analysis </h3>
TBA
<h3> Topic Analysis </h3>
TBA
<h3> Summarization </h3>
TBA

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


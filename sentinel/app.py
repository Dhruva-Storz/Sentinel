from flask_socketio import SocketIO, emit

from flask import Flask, render_template, url_for, request, redirect, Blueprint, copy_current_request_context, jsonify
import os
import time
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
import sys

from random import random
from time import sleep
from threading import Thread, Event
import threading
import argparse


import multiprocessing
multiprocessing.set_start_method('spawn', True) # makes CUDA compatible with multiprocessing
import concurrent.futures

from scraping.crawler import TwitterClient
from topic_analysis.topicanalysis import get_topics
from main_SA import SentimentAnalysisModel
from main_Sum import TweetSummariser

client = TwitterClient()

app = Flask(__name__)

socketio = SocketIO(app, async_mode=None, logger=False, engineio_logger=False)

thread = Thread()
thread_stop_event = Event()
end_thread = False
result_type = "recent"
num_tweets = 3000
num_topics = 6

tweets = []
keyword = ''

model_path = '/mnt/c/Users/Dhruva/Desktop/Projects/msc-ai-group-project-master/msc-ai-group-project-master/sentinel/sentiment_analysis/saved_model_new/bert-base-uncased_final_1.pt'

SA = SentimentAnalysisModel(model_path=model_path, debug=False)

# Summariser = TweetSummariser(debug=True,
#                              bert_pretrained_model='bert-base-uncased',
#                              bert_pool='avg',
#                              bert_use_hidden=True,
#                              bert_max_len=300,
#                              bert_batch_size=16,
#                              use_sbert=True,
#                              dist_metric='cosine',
#                              pca_components=None,
#                              bart_pretrained_model='semsim',
#                              filter_sentiment=True,
#                              softmax_threshold_neg=0.98,
#                              softmax_threshold_pos=0.98,
#                              summary_max_len=70
#                              )


def run_analysis(keyword, result_type, num_tweets, num_topics):
    if not keyword:
        keyword = 'cat'
    tstart = time.time()
    df, samples = client.get_tweets_for_keyword(keyword, count=50)
   
    socketio.emit('keyword', {'keyword': keyword}, namespace = '/test')
    socketio.emit('status', {'status':'Fetching Tweets'}, namespace = '/test')
    socketio.emit('loading', {'tweet':samples}, namespace = '/test')


    # Fetch Tweets
    t0 = time.time()
    df, tweets = client.get_tweets_for_keyword(keyword, count=num_tweets, result_type=result_type)
    print(f"Tweet fetching: {time.time()-t0} (s)")

    socketio.emit('status', {'status':'Analysing'}, namespace = '/test')


    # Perform Analysis
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor: # Threadpool is faster than ProcessPool
        #f2 = executor.submit(get_topics, keyword, tweets, num_topics=8, no_below = 0, no_above=0.75, num_passes=10)
        f2 = executor.submit(get_topics, keyword, tweets, multicore=False, preproc_old=False, num_topics=num_topics)
        f1 = executor.submit(SA.return_sentiment, df, tweet_col='tweets', batch_size=32, emoji_voting=True, apply_softmax=True, plot_sa=False, plot_path='static/SA_plot.png', plot_n=num_topics)
        plt.close('all')

    print(f'SA and TA: {time.time()-t0} (s)')

    t0 = time.time()
    topiclist, topicstrings, df_dominant_topic = f2.result()
    sa_df, SA_str = f1.result()
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    # print(sa_df)

    socketio.emit('results', {'sentiment': SA_str, 'topic': topicstrings, 'summary': []}, namespace='/test')
    print(time.time()-t0)

    t0 = time.time()
    summaries = Summariser.return_summary(sa_df,
                                          col_tweet='tweets',
                                          col_sentiment='sentiment',
                                          col_softmax='softmax',
                                          k=num_topics,
                                          n_sent=2)

    print(f'Summarization: {time.time()-t0} (s)')

    print(summaries)

    print(f"total time: {time.time()-tstart} (s)")

    socketio.emit('results', {'sentiment': SA_str, 'topic': topicstrings, 'summary': summaries}, namespace='/test')
    socketio.emit('status', {'status':'Finished'}, namespace = '/test')

@app.route('/', methods=['GET','POST'])
def index():

    if request.method == 'POST':
        # # try:

        global keyword
        # Retrieve user input from html form
        keyword = request.form["input"]

        if keyword == '':
            return render_template('index.html'), 400

        global result_type
        result_type = "recent"

        global num_tweets
        num_tweets = 3000

        global num_topics
        num_topics = 6

        return render_template('d3.html', text=keyword.upper()), 200
    else:
        return render_template('index.html'), 200



@app.route('/advanced', methods = ['GET', 'POST'])
def advanced():
    if request.method == 'POST':
        global num_tweets
        global keyword
        global tweet_type
        global num_topics

        keyword = request.form["advinput"]
        if keyword == '':
            return render_template('advanced.html'), 400
        num_tweets = int(request.form["num_tweets"])
        tweet_type = request.form["tweet_type"]
        num_topics = int(request.form["num_topics"])
        if not keyword:
            keyword = 'cat'


        return render_template('d3.html', text=keyword.upper()), 200
        # except:
        #     print("There was a problem")
        #     return render_template('advanced.html')
    else:
        return render_template('advanced.html')

@app.route('/get-data', methods = ['GET', 'POST'])
def return_data():
    with open("./static/files/SA_time_binned.json", 'r') as f:
        data = f.read()
    return data

@app.route('/get-data-italian-flag', methods = ['GET', 'POST'])
def return_data_italian_flag():
    with open('./static/files/italian_flag.json', 'r') as f:
        data = f.read()
    return data

@app.route('/get-data-sa-json', methods = ['GET', 'POST'])
def return_sa_json():
    with open("./static/files/sa_d3.json", 'r') as f:
        data = f.read()
    return data

@app.route('/get-data-sa-idx', methods = ['GET', 'POST'])
def return_sa_idx():
    with open("./static/files/sa_idx.json", 'r') as f:
        data = f.read()
    return data

@socketio.on('connect', namespace='/test')
def test_connect():
    # need visibility of the global thread object
    global thread
    print(len(threading.enumerate()), file=sys.stderr)

    thread = Thread()
    print('Client connected', file=sys.stderr)
    global keyword
    global result_type
    global num_tweets
    global num_topics


    #Start the random number generator thread only if the thread has not been started before.
    if not thread.isAlive():
        print("Starting Thread")
        thread = socketio.start_background_task(run_analysis, keyword=keyword, result_type=result_type, num_tweets=num_tweets, num_topics=num_topics)

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure the Flask app.')
    parser.add_argument("-p", "--port", type=int, default=5000,
                        help="Port number")
    args = parser.parse_args()
    socketio.run(app, host='0.0.0.0', port=args.port)

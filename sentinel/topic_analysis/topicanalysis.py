import datetime
import pandas as pd, numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import nltk
import re
from gensim import corpora, models
from gensim.utils import lemmatize, simple_preprocess
from topic_analysis.preprocessing_lda import get_wordnet_pos, clean_tweets, sent_to_words, process_words
import matplotlib.pyplot as plt
from spacy import load
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, FancyBboxPatch

import matplotlib as mpl
from matplotlib.transforms import Bbox

import pickle
import json
import datetime


# move things like this to init file for final release
nltk.download('averaged_perceptron_tagger', quiet=True) #quiet removes those uneccesary print things
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


def get_topics(usertopic, tweets, num_topics=6, num_words=7,
                num_passes=20, top_words=15, no_below=0, no_above=0.95,
                multicore=False, preproc_old=False, threshold=0.3#0.1#100
                ):
    # Identify topic:
    topic = usertopic
    # Remove topic (lemmatize and make it a stopword):
    topic_stop_word = [WordNetLemmatizer().lemmatize(topic.lower(), get_wordnet_pos(topic))]
    topic_stop_word.append(topic.lower().split())

    if preproc_old:
        # populate the corpus with tokens (version 1.0)
        corpus = []
        for tweet in tweets:
            corpus.append(clean_tweets(tweet, topic_stop_word, bigrams=True))
            corpus.append(clean_tweets(tweet, topic_stop_word, bigrams=False))
        token_tweets = corpus

    else:
        data_words = list(sent_to_words(tweets))
        # Build the bigram and trigram models
        bigram = models.Phrases(data_words, min_count=5, threshold=threshold) # higher threshold fewer phrases.
        trigram = models.Phrases(bigram[data_words], threshold=threshold)
        bigram_mod = models.phrases.Phraser(bigram)
        trigram_mod = models.phrases.Phraser(trigram)

        token_tweets = process_words(data_words, bigram_mod, trigram_mod, topic_stop_word)
        # print('CHECK THE CORPUS: ', token_tweets)


    tweets_dict = corpora.Dictionary(token_tweets)
    #tweets_dict.filter_extremes(no_below=no_below, no_above=no_above)

    bow_corpus = [tweets_dict.doc2bow(doc) for doc in token_tweets]

    tfidf = models.TfidfModel(bow_corpus)
    tfidf_corpus = tfidf[bow_corpus]

    NUM_TOPICS = num_topics # <-- EDIT: NUMBER OF TOPICS
    NUM_PASSES = num_passes # 50 used in example, 10-20 should work too
    ALPHA = 0.3 #0.001 # Each tweet belongs to one or two topics, not more... (OG: 0.001)
    ETA = 'auto'
    # Train LDA model

    if multicore:
        # workers should default to total - 1, if not set manually below:
        lda_model = models.ldamulticore.LdaMulticore(corpus=bow_corpus,
                                        num_topics=NUM_TOPICS,
                                        id2word=tweets_dict,
                                        passes=NUM_PASSES,
                                        alpha=ALPHA,
                                        eta=ETA,
                                        random_state=49,
                                        per_word_topics=True)
    else:
        lda_model = models.ldamodel.LdaModel(corpus=bow_corpus,
                                        num_topics=NUM_TOPICS,
                                        id2word=tweets_dict,
                                        passes=NUM_PASSES,
                                        alpha=ALPHA,
                                        eta=ETA,
                                        random_state=49,
                                        per_word_topics=True)


    corpus_lda = lda_model[tfidf_corpus]

    topics = lda_model.show_topics(NUM_TOPICS,top_words) # <-- OUTPUT as list of tuples [(0,'0.075*"ricky" + 0.072*"gervais" + 0.021*"rickygervais"'),
                                        #                                 (1, '0.075*"ricky" + 0.072*"gervais" + 0.021*"rickygervais")
    ########################
    ### OLD OUTPUT #########
    ########################

    topiclist = []
    topicstrings = []
    c = 1
    for tup in topics:
        pseudowordlist = tup[1].split(' + ')
        wordlist = []
        for item in pseudowordlist:
            word = item.split('*')
            word = word[1]
            word = word.replace('"', "")
            wordlist.append(word)
        topiclist.append(wordlist[:num_words])
        topicstrings.append('Topic ' + str(c) + ': ' + str(wordlist[:num_words]))
        c += 1

    #########################
    ###### NEW OUTPUT #######
    #########################

    # Create a pandas dataframe with a single row per tweet
    # Doc_num: simply an index (i.e., number of tweet in this dataframe)
    # Dominant_topic: which topic does the tweet mainly belong to
    # Topic percent contribution
    # Keywords: Most representative tokens in the topic given by Dominant_topic
    # Text: tokenized tweet


    # Plot TA chart
    sentences_chart(lda_model, tfidf_corpus, start=0, end=5, n_topic=num_topics)

    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=bow_corpus, texts=token_tweets)
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    ###### CAN GET MOST REPRESENTATIVE TWEET PER TOPIC WITH THIS CODE #########
    # sent_topics_sorteddf_mallet = pd.DataFrame()
    # sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
    #
    # for i, grp in sent_topics_outdf_grpd:
    #     sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
    #                                              grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],
    #                                             axis=0)
    #
    # # Reset Index
    # sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
    #
    # # Format
    # sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]


    ############### Wordcloud ###############
    plot_wordcloud(lda_model)
    print('INFO: LDA done at ', datetime.datetime.now())

    return topiclist, topicstrings, df_dominant_topic


def plot_wordcloud(lda_model):
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    x, y = np.ogrid[:300, :300]

    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)

    cloud = WordCloud(stopwords=[],
                      background_color="rgba(255, 255, 255, 0)", mode="RGBA",
                      width=3000, #2500
                      height=3000, #1800
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=0.5,#1.0
                      mask=mask
                      )

    topics = lda_model.show_topics(formatted=False)

    num_topics = len(topics)
    n_cols = 3
    n_rows = num_topics // n_cols if not num_topics%n_cols else num_topics // n_cols + 1
    fig = plt.figure(figsize=(12, int(12/n_cols*n_rows)))
    gs = mpl.gridspec.GridSpec(n_rows, n_cols, wspace=0.05, hspace=0)
    if num_topics%n_cols:
        missing = n_cols - num_topics%n_cols
        frac = (missing / n_cols) / 2
        gs2 = mpl.gridspec.GridSpec(n_rows, n_cols*2, wspace=0.05, hspace=0)

    axes = []
    for i in range(num_topics):
        if i < n_cols*(n_rows-1) or num_topics == n_cols*n_rows:
            axes.append(gs[i//n_cols, i%n_cols])
        else:
            axes.append(gs2[i//n_cols, missing+(i%n_cols)*2:missing+(i%n_cols+1)*2])

    for i, ax in enumerate(axes):
        ax = fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().axis('off')

    plt.axis('off')
    plt.savefig('static/wordcloud.png', transparent=True, dpi=300,
                bbox_inches=Bbox([[1.6, .4 + .32*n_rows], [10.7, int(12/n_cols*n_rows)-.4 -.32*n_rows]]))


# Sentence Coloring of N Sentences
def sentences_chart(lda_model, corpus, start=0, end=13, n_topic=None):
    if not n_topic:
        end += 1
        corp = []
        for c in corpus:
            if c:
                corp.append(c)
            if len(corp) == end - start:
                break
    else:
        corp = corpus
    # corp = corpus[start:end]

    mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    n_chart = n_topic + 1 if n_topic else (end - start)
    height = n_chart * 1.7
    fig, axes = plt.subplots(n_chart, 1, figsize=(18, height), dpi=200)
    r = fig.canvas.get_renderer()

    plot_idx = []
    n_tweet = 0
    plotted_topic = set()
    axes[0].axis('off')
    for idx, corp_cur in enumerate(corp):
        if n_tweet == n_chart - 1:
            break
        if not corp_cur:
            continue
        topic_percs, wordid_topics, wordid_phivalues = lda_model[corp_cur]
        word_dominanttopic = [(lda_model.id2word[wd], topic[0]) for wd, topic in wordid_topics]
        topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
        if topic_percs_sorted[0][0] in plotted_topic and n_topic is not None:
            continue
        n_tweet += 1
        plotted_topic.add(topic_percs_sorted[0][0])
        plot_idx.append(idx)
        ax = axes[n_tweet]
        ax.axis('off')
        inv = ax.transData.inverted()
        ax.text(0.015, 0.5, "Tweet " + str(n_tweet) + ": ", verticalalignment='center',
                fontsize=20, color='white', transform=ax.transAxes, fontweight=700)
        ax.add_patch(FancyBboxPatch((0.002, 0.07),
                    0.98, 0.90,
                    boxstyle="round,pad=-0.0040,rounding_size=0.015",
                    # ec="none",
                    mutation_aspect=14,
                    fill=None,# alpha=1,
                    color=mycolors[topic_percs_sorted[0][0]], linewidth=4
                    ))
        word_pos = 0.13
        for j, (word, topics) in enumerate(word_dominanttopic):
            if j < 14:
                txt = ax.text(word_pos, 0.5, word,
                              horizontalalignment='left',
                              verticalalignment='center',
                              fontsize=20,
                              color=mycolors[topics],
                              transform=ax.transAxes,
                              fontweight=700,
                              # bbox=dict(facecolor='none', edgecolor='red')
                              )

                bb = txt.get_window_extent(renderer=r)
                coord0, coord1 = bb.get_points()
                x0, y0 = inv.transform(coord0)
                x1, y1 = inv.transform(coord1)

                if x1 > 0.92:
                    txt.remove()
                    break
                word_pos = x1 + .02
                # word_pos += .009 * len(word)  # to move the word for the next iter
                ax.axis('off')

        ax.text(word_pos, 0.5, '. . .',
                horizontalalignment='left',
                verticalalignment='center',
                fontsize=20, color='white',
                transform=ax.transAxes)

    plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.savefig('static/TA_chart.png', transparent=True, dpi=200,
                bbox_inches=Bbox([[1.8, .5], [16.5, height - 1.3]]))

    with open("./static/files/sa_idx.json", 'w') as file:
        json.dump(plot_idx, file)


# tweets = ["Arnab is the ass of Assam...", "*On tour* Today’s Kuwait show sold out, Singapore. Melbourne & Sydney second show has been added... Tickets on http://kunalkamra.in Baaki dekhlo... tell your friends make a plan! Maro RT", "Congratulations to Eddie DeBartolo Jr. and your wonderful family of friends!", "...And this despite Fake Witch Hunts, the Mueller Scam, the Impeachment Hoax etc. With our Economy, Jobs, Military, Vets, 2A & more, I would be at 70%. Oh well, what can you do?" , "I hope the Federal Judges Association will discuss the tremendous FISA Court abuse that has taken place with respect to the Mueller Investigation Scam, including the forging of documents and knowingly using the fake and totally discredited Dossier before the Court. Thank you!"]

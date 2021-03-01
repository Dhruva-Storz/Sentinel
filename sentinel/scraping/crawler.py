
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream, API, Cursor
import scraping.twittercredentials as creds
import re
import pandas as pd
import numpy as np

class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client

    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet.full_text for tweet in tweets], columns=["tweets"])
        # df['id'] = np.array([tweet.id for tweet in tweets])
        # df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        # df['source'] = np.array([tweet.source for tweet in tweets])
        # df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])
        # df['length'] = np.array([len(tweet.text) for tweet in tweets])
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        return df


    def get_tweets_for_keyword(self, keyword, count=100, result_type="recent"):

        keyword = keyword + " -filter:retweets"
        
        tweets = Cursor(self.twitter_client.search, q=keyword, count=count, lang = 'en', tweet_mode='extended', result_type=result_type).items(count)

        # for tweet_info in Cursor(self.twitter_client.search, q=keyword, count=count, lang = 'en', tweet_mode='extended').items(count):
        #     if 'retweeted_status' in dir(tweet_info):
        #         tweet=tweet_info.retweeted_status.full_text
        #     else:
        #         tweet=tweet_info.full_text
        #     tweets.append(tweet)

        dates = []
        tweets = []

        for tweet in Cursor(self.twitter_client.search, q=keyword, count=count, lang = 'en', tweet_mode='extended').items(count):
            dates.append(tweet.created_at)
            tweets.append(tweet.full_text)

        df = pd.DataFrame(data=np.array(tweets), columns=["tweets"])
        df['date'] = np.array(dates)

        return df, tweets


class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = OAuthHandler(creds.CONSUMER_KEY, creds.CONSUMER_SECRET)
        auth.set_access_token(creds.ACCESS_TOKEN, creds.ACCESS_TOKEN_SECRET)
        return auth


if __name__ == "__main__":
    client = TwitterClient()
    c = client.get_twitter_client_api()
    query = "oscars"
    tweets = client.get_tweets_for_keyword(query, count=100 )
    # tweets = Cursor(c.search, q = query, count = 10,lang = 'en', tweet_mode='extended').items(10)
    #iD = [tweet.id for tweet in tweets]
    print(tweets)

import pandas as pd
import re


class SummarisationPreprocessor(object):
    def __init__(self,
                 min_len=10,
                 keep_max=True,
                 remove_username=True,
                 keep_fullstop=True,
                 not_start_words=["but", "and", "or", "i", "i'm", "i’m",
                                  "im", "i've", "i’ve", "ive", "i'll",
                                  "i’ll" , "i'd", "i’d", "id", "my", "we",
                                  "we've", "we’ve", "weve", "we'll",
                                  "we’ll", "we'd", "we’d", "wed", "our",
                                  "you", "he", "she", "they", "this",
                                  "that", "these", "those", "why", "what",
                                  "who", "which", "when"],
                 not_contain_words=["i", "i'm", "i’m", "im", "i've",
                                    "i’ve", "ive", "i'll", "i’ll" , "i'd",
                                    "i’d", "me", "my", "we", "our", "you",
                                    "your", "fuck", "shit"]
                 ):
        """
        Preprocess the list of tweets by removing any url, emoji, punctuation.
        It also splits the sentences in each tweet and only keep sentences
        longer than `min_len`. By default, it'll return a single longest
        sentence in each tweet.

        :param content: list of tweets
        :param min_len: minimum length of a sentence to keep in tweets
        :param keep_max: if True, only keep the longest sentence in the tweet
        :param remove_username: if True, remove the entire username starting
                with @. Otherwise, only remove '@' and keep the username
        :param keep_fullstop: if True, only keep sentences that end with a full
                stop i.e. removes any sentence that ends with '?' or '!'
        :param not_start_words: list of words the sentence should NOT start
                with. Make sure the words are all lowercased.
        :param not_contain_words: list of words the sentence should NOT contain.
                Make sure the words are all lowercased

        :return: list of preprocessed tweets
        """
        self.min_len = min_len
        self.keep_max = keep_max
        self.remove_username = remove_username
        self.keep_fullstop = keep_fullstop

        self.punctuation_pattern = r'[`"“”\#$%&()*+\-/;<=>\[\\\]^_{|}~]'

        self.not_startwith = r' |'.join(not_start_words) + ' '
        self.not_contain = ' ' + r' | '.join(not_contain_words) + ' '
        if not self.not_startwith.strip():
            self.not_startwith = 'ALL STARTS VALID'
        if not self.not_contain.strip():
            self.not_contain = 'ALL WORDS VALID'

    def preprocess_tweets(self, content):
        tweets = []
        dup_tweet = set() # Use set so it exludes any duplicates
        for tweet in content:
            # Make tweet into one line with each line considered as a sentence
            tweet = re.sub("(?<=[\.\!\?])\s*\n\s*", ' ', tweet)
            tweet = re.sub('\s*\n\s*', '. ', tweet)
            tweet = tweet.encode('ascii', 'ignore').decode('ascii') # Remove emojis
            tweet = re.sub('http\S+', '', tweet) # Get rid of urls
            # tweet = re.sub('\?.', '. ', tweet)
            # tweet = re.sub('\!.', '. ', tweet)
            tweet = re.sub(self.punctuation_pattern, '', tweet) # Remove punctuations
            tweet = re.sub("\:", '. ', tweet)
            tweet = re.sub(r'\bamp\b', ' ', tweet) # Get rid of 'amp' only if it's not part of a word e.g. amplify

            # Handle the usernames
            if self.remove_username: # remove '@' AND the following name
                tweet = re.sub(r'@\S+', '', tweet)
            else: # only remove '@' symbol and keep the actual username
                tweet = re.sub("@", '', tweet)

            # Reduce any whitespace of length > 1 to one
            tweet = re.sub(r'\s\s+', ' ', tweet)

            cleaned_tweet = []
            # Split sentences ending with '.', '!', '?'
            for sent in re.split(r'(?<=[\.\!\?])\s+', tweet):
                sent = sent.strip()
                if self.keep_fullstop and (sent.endswith('?') or sent.endswith('!') or sent.endswith('!.') or sent.endswith('?.')):
                    continue
                if len(sent.split()) > self.min_len \
                   and not re.match(self.not_startwith, sent.lower()) \
                   and not re.search(self.not_contain, sent.lower()):
                    if sent[-1] not in ['.', '!', '?']:
                        sent += '.'
                    cleaned_tweet.append(sent)
            if len(cleaned_tweet) == 0:
                continue

            if self.keep_max: # Only keep the longest sentence in the tweet
                cleaned_tweet = max(cleaned_tweet)
            else:
                cleaned_tweet = ' '.join(cleaned_tweet)

            if cleaned_tweet[:50].lower() not in dup_tweet:
                tweets.append(cleaned_tweet)
                dup_tweet.add(cleaned_tweet[:50].lower())

        return list(tweets)

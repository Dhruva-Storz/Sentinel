from unittest import TestCase
from preprocessing import Preprocessor, UniversalPreprocessor, SentimentPreprocessor, LDAPreprocessor, SummarisationPreprocessor

corpus1 = ["Sentances, @wooord, 😄, 🦋💯", "short 😨", "this is the longest sentences in the corpus https://twitter.com."]
corpus2 = ["and this is why tomato is a vegetable",
           "short tweet",
           "NEWS: Sentances, @wooooord, 😄, 🦋💯",
           "short 😨",
           "This is a shorter sentence. this is the longest sentences in the tweet https://twitter.com."]

class TestPreprocessor(TestCase):
    def test_preprocessed_corpus(self):
        prep1 = Preprocessor(corpus1)
        # Test whether the first sentence has been preprocessed correctly
        corpus = prep1.preprocessed_corpus()[0]
        self.assertEqual(corpus[0], ["sentances", "word", "<positiveemoji>", ":butterfly:", "<positiveemoji>"])
        # Test whether the second sentence has been removed as it should because it is so short
        # And whether the last sentence is preprocessed correctly
        self.assertEqual(corpus[1], ['this', 'is', 'the', 'longest', 'sentences', 'in', 'the', 'corpus'])
        self.assertEqual(len(corpus), 2)
        self.assertEqual([0, 2], prep1.preprocessed_corpus()[1])
        prep1 = Preprocessor(corpus1, tokenize=False)
        corpus = prep1.preprocessed_corpus()[0]
        # Test whether the first sentence has been preprocessed correctly
        self.assertEqual(corpus[0], "sentances, word, <positiveemoji> :butterfly: <positiveemoji>")
        # Test whether the second sentence has been removed as it should because it is so short
        # And whether the last sentence is preprocessed correctly
        self.assertEqual(corpus[1], 'this is the longest sentences in the corpus')
        self.assertEqual((len(corpus)), 2)
        self.assertEqual([0, 2], prep1.preprocessed_corpus()[1])
        prep1 = Preprocessor(corpus2)
        corpus = prep1.preprocessed_corpus()[0]
        self.assertEqual(['sentances', 'word', '<positiveemoji>', ':butterfly:', '<positiveemoji>'], corpus[2])



    def test_spelling_correction(self):
        prep2 = Preprocessor(corpus1, spellcorrect=True)
        corpus = prep2.preprocessed_corpus()[0]
        self.assertEqual(corpus[0], ["sentences", "word", "<positiveemoji>", "butterfly", "<positiveemoji>"])

    def test_lemmatization(self):
        prep3 = Preprocessor(corpus1, lemmatize=True)
        corpus = prep3.preprocessed_corpus()[0]
        self.assertEqual(corpus[1], ['this', 'be', 'the', 'longest', 'sentence', 'in', 'the', 'corpus'])

    def test_emoji_replacement(self):
        prep4 = Preprocessor(corpus1, remove_short_tweets=False, verbose_emoji=True)
        corpus = prep4.preprocessed_corpus()[0]
        self.assertEqual(corpus[1], ['short', ':fearful_face:'])

    def test_get_emoji_score(self):
        prep5 = Preprocessor(corpus1, remove_short_tweets=False, verbose_emoji=False)
        self.assertEqual(prep5.get_emoji_score(0)["positive"], 2)
        self.assertEqual(prep5.get_emoji_score(0)["negative"], 0)
        self.assertEqual(prep5.get_emoji_score(1)["positive"], 0)
        self.assertEqual(prep5.get_emoji_score(1)["negative"], 1)
        self.assertEqual(prep5.get_emoji_score(2)["positive"], 0)
        self.assertEqual(prep5.get_emoji_score(2)["negative"], 0)
        prep5 = Preprocessor(corpus1, remove_short_tweets=False, verbose_emoji=True)
        self.assertEqual(prep5.get_emoji_score(0)["positive"], 2)
        self.assertEqual(prep5.get_emoji_score(0)["negative"], 0)
        self.assertEqual(prep5.get_emoji_score(1)["positive"], 0)
        self.assertEqual(prep5.get_emoji_score(1)["negative"], 1)
        self.assertEqual(prep5.get_emoji_score(2)["positive"], 0)
        self.assertEqual(prep5.get_emoji_score(2)["negative"], 0)


    def test_reduce_repeated_letters_corpus(self):
        prep6 = Preprocessor(["Teeest test"], remove_short_tweets=False, reduce_chars=True)
        corpus = prep6.preprocessed_corpus()[0]
        self.assertEqual(corpus[0], ["test", "test"])
        prep6 = Preprocessor(["Teeest test"], remove_short_tweets=False, reduce_chars=False)
        corpus = prep6.preprocessed_corpus()[0]
        self.assertEqual(corpus[0], ["teeest", "test"])

    def test_has_proper_noun(self):
        prep7 = Preprocessor([])
        self.assertTrue(prep7.has_proper_noun("Bernie will win."))
        self.assertFalse(prep7.has_proper_noun("not impeachable!"))
        self.assertFalse(prep7.has_proper_noun("🤕"))


    def test_universalPreprocessor(self):
        prep8 = UniversalPreprocessor(corpus1)
        self.assertEqual(prep8.corpus[0], ['sentances', 'word', '😄', '🦋💯'])
        self.assertEqual(prep8.corpus[1], ['short', '😨'])
        self.assertEqual(prep8.corpus[2], ['this', 'is', 'the', 'longest', 'sentences', 'in', 'the', 'corpus'])


    def test_sentimentPreprocessor(self):
        prep9 = SentimentPreprocessor(corpus1)
        corpus = prep9.preprocessed_corpus()[0]
        self.assertEqual('sentances, word, <positiveemoji> :butterfly: <positiveemoji>', corpus[0])
        self.assertEqual('this is the longest sentences in the corpus', corpus[1])
        self.assertEqual(2, len(corpus))
        self.assertEqual([0, 2], prep9.preprocessed_corpus()[1])

    def test_LDAPreprocessor(self):
        stopwords = ["word", "this"]
        prep10 = LDAPreprocessor(corpus1, stopwords=stopwords)
        corpus = prep10.preprocessed_corpus()[0]
        self.assertEqual(['sentances', ':grinning_face_with_smiling_eyes:', ':butterfly:', ':hundred_points:'], corpus[0])
        self.assertEqual(['longest', 'sentence', 'corpus'], corpus[1])

    def test_summarisationPreprocessor(self):
        prep11 = SummarisationPreprocessor(corpus2)
        corpus = prep11.preprocessed_corpus()[0]
        indices = prep11.preprocessed_corpus()[1]
        self.assertEqual([1, 2, 4], indices)
        self.assertEqual(['news', 'sentances', 'word', '😄', '🦋💯'], corpus[1])
        self.assertEqual(['this', 'is', 'the', 'longest', 'sentences', 'in', 'the', 'tweet'], corpus[2])
        print(corpus)
        pass


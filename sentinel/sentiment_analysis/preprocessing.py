from spellchecker import SpellChecker
from nltk.stem.wordnet import WordNetLemmatizer
import re
import spacy
import emoji

class Preprocessor:
	"""
	Performs preprocessing of tweets.
	"""

	TAGS = {"@", "#"}

	ALL_EMOJIS = emoji.unicode_codes.UNICODE_EMOJI
	POSITIVE_EMOJIS = {e for e in
					   "😀♥️❤️💙💚💜😃💫⭐️🌟💯😄😁☺️😊😇🙂🙃😉😌😍🥰😘😗😙😚🥰😛😜😎🤩🥳😏🤗🤠😹😺😸😻😽👏🏻👍🏻✌️💪🏻"}
	NEGATIVE_EMOJIS = {e for e in "😞😔😟😕🙁☹️😣😖😫😩😢😭😤😠😡🤬😨😰😥😓🤒🤕🤢🤮😾😿👎🖕🏻"}

	STOPWORDS = ["NEWS", "news", "News", "RT"]
	def __init__(self, corpus: list,
				 tokenize=True,
				 emoji_handling=True,
				 verbose_emoji=False,
				 reduce_chars=True,
				 url_removal=True,
				 tag_removal=True,
				 remove_short_tweets=True,
				 spellcorrect=False,
				 lemmatize=False,
				 remove_short_words=False,
				 keep_only_longest_sentence=False,
				 stopwords = STOPWORDS):
		"""
		:param corpus: Corpus (list of strings (e.g. tweets)) to be preprocessed
		"""
		verbose_emoji = emoji_handling and verbose_emoji
		super().__init__()
		self.spacy = spacy.load("en_core_web_sm")
		self.corpus = corpus
		self.idx = list(range(len(self.corpus)))
		self.return_tokenized = tokenize
		#self.emoji_replacement_string_sentences(verbose=verbose_emoji)
		self.tokenize(url_removal=url_removal,
					  tokenize=tokenize,
					  tag_removal=tag_removal,
					  remove_short_tweets=remove_short_tweets,
					  remove_short_words=remove_short_words,
					  stopwords=stopwords,
					  keep_only_longest_sentence=keep_only_longest_sentence)
		self.emoji_scores = {tweet: {"positive": 0, "negative": 0} for tweet in range(len(self.corpus))}
		self.preprocess(spellcorrect=spellcorrect,
						lemmatize=lemmatize,
						emoji_handling=emoji_handling,
						reduce_chars=reduce_chars,
						verbose_emoji=verbose_emoji)
		self.corpus = Preprocessor.corpus_remove_stopwords(self.corpus, stopwords)



	def preprocessed_corpus(self):
		if not self.return_tokenized:
			corpus = [" ".join(tweet) for tweet in self.corpus]
		else:
			corpus = self.corpus
		return corpus, self.idx

	def tokenize(self,
				 tokenize=True,
				 url_removal=True,
				 tag_removal=True,
				 remove_short_tweets=True,
				 remove_short_words=False,
				 stopwords=[],
				 keep_only_longest_sentence=False):
		"""
		Tokenizes and lowercases the corpus, i.e. turns it into a list of lists,
		where each list contains the individual words of the strings in the corpus, but lowercased and with punctuation
		removed.
		If url_removal is True, urls are removed.
		If tag_removal is True, # and @ are removed.
		"""
		re_punctuation_string = '[]'
		if tokenize:
			re_punctuation_string = '[\s,/.?!:;]'
			if tag_removal:
				re_punctuation_string = '[\s,/.?!@#:;]'
		else:
			if tag_removal:
				re_punctuation_string = '[\s@#]'
		tokenized_corpus = []

		# If a tweet is to short and has no proper noun -> remove it
		for idx, sentence in enumerate(self.corpus):
			if remove_short_tweets and len(sentence.split()) <= 3:
				if not self.has_proper_noun(sentence):
					self.idx.remove(idx)
					continue
			# Remove urls
			if url_removal:
				sentence = " ".join(filter(lambda x:
													x and x[:7] != "http://" and x[:8] != "https://" and x[:4] != "amp.",
													sentence.split()))
			# If we only want to keep the longest sentence, remove all others here
			if keep_only_longest_sentence:
				sentences = re.split('[!?.]', sentence)
				sentence = max(sentences, key = lambda x: len(x))
			tokenized_sentence_PR = re.split(re_punctuation_string, sentence)
			tokenized_sentence_PR = list(filter(lambda x: x and
														  x not in stopwords
														  and (x[0] in Preprocessor.ALL_EMOJIS or not (remove_short_words and len(x) < 4)),
												tokenized_sentence_PR))  # remove empty strings from list
			#lowercase:
			tokenized_sentence_PR = [x.lower() for x in tokenized_sentence_PR]

			tokenized_corpus.append(tokenized_sentence_PR)
		self.corpus = tokenized_corpus


	def spelling_correction(self):
		"""
		Corrects the spelling of all words in the corpus (e.g. "hii -> hi")
		"""

		# TODO:
		# Check whether it is only slow if it has lots of things to correct
		# Check whether it can be sped up by GPU
		# Check whether running in parallel increases speed
		# Check whether there are faster libraries


		spell = SpellChecker()
		for i, sentence in enumerate(self.corpus):
			for j, word in enumerate(sentence):
				word = spell.correction(word)
				self.corpus[i][j] = word


	def lemmatization(self):

		for i, sentence in enumerate(self.corpus):
			for j, word in enumerate(sentence):
				self.corpus[i][j] = Preprocessor.lemmatize(word)


	# def emoji_replacement_string_sentences(self, verbose=False):
	# 	"""
	# 	Replace certain emojis (e.g. 😊 or 🙂) with <positive>, others (e.g. 😞 or 😭) with <negative>. Emojis that
	# 	are neither clearly positive nor clearly negative are replaced by a tag describing them.
	# 	:param verbose:If this is true, each emoji is replaced by a tag describing it explicitly. E.g. '😉' would be
	# 	replaced by ':winking_face:'
	# 	:return:
	# 	"""
	# 	for i, sentence in enumerate(self.corpus):
	# 		sentence = sentence.split()
	# 		for j, word in enumerate(sentence):
	# 			if not word[0] in Preprocessor.ALL_EMOJIS:
	# 				continue
	# 			emojis = [e for e in word if e in Preprocessor.ALL_EMOJIS]
	# 			replacements = [Preprocessor.replace_emojis(e) for e in emojis]
	# 			self.emoji_scores[i]["positive"] += replacements.count("<positiveemoji>")
	# 			self.emoji_scores[i]["negative"] += replacements.count("<negativeemoji>")
	# 			if verbose:
	# 				replacements = [emoji.demojize(e) for e in emojis]
	# 			else:
	# 				replacements = [emoji.demojize(e) for e in replacements]
	# 			sentence = sentence[:j] + replacements + sentence[j+1:]
	# 		sentence = " ".join(sentence)
	# 		self.corpus[i] = sentence
	# 			#self.corpus[i].insert(j, replacements)

	def emoji_replacement(self, verbose=False):
		"""
		Replace certain emojis (e.g. 😊 or 🙂) with <positive>, others (e.g. 😞 or 😭) with <negative>. Emojis that
		are neither clearly positive nor clearly negative are replaced by a tag describing them.
		:param verbose:If this is true, each emoji is replaced by a tag describing it explicitly. E.g. '😉' would be
		replaced by ':winking_face:'
		:return:
		"""
		for i, sentence in enumerate(self.corpus):
			for j, word in enumerate(sentence):
				if not word[0] in Preprocessor.ALL_EMOJIS:
					continue
				emojis = [e for e in word if e in Preprocessor.ALL_EMOJIS]
				replacements = [Preprocessor.replace_emojis(e) for e in emojis]
				self.emoji_scores[i]["positive"] += replacements.count("<positiveemoji>")
				self.emoji_scores[i]["negative"] += replacements.count("<negativeemoji>")
				if verbose:
					replacements = [emoji.demojize(e) for e in emojis]
				else:
					replacements = [emoji.demojize(e) for e in replacements]
				self.corpus[i] = self.corpus[i][:j] + replacements + self.corpus[i][j+1:]
				#self.corpus[i].insert(j, replacements)



	def get_emoji_score(self, tweet_number:int)->dict:
		"""
		Gives the emoji-score of a tweet. The emoji-score is a dictionary that records the number
		of positive and negative emojis that appear in the tweet
		:param tweet_number: the index of the tweet in question in the corpus
		:return: a dictionary {"positive": n, "negative": m} where n is the number of positive emojis and m is
		the number of negative emojis in the tweet
		"""
		return self.emoji_scores[tweet_number]





	def reduce_repeated_letters_corpus(self):
		"""
		Replace substrings consisting of multiple repetitions of the same character "hellooooo" with just two instances
		# of that character "helloo"
		:return:
		"""
		reduced_corpus = []
		for i, sentence in enumerate(self.corpus):
			reduced_sentence = []
			for j, word in enumerate(sentence):
				self.corpus[i][j] = Preprocessor.reduce_repeated_letters(word)


	def has_proper_noun(self, sentence:str)->bool:
		"""
		Determines whether the given sentence contains a proper noun
		"""

		spacy_analysis = self.spacy(sentence)
		return "PROPN" in [token.pos_ for token in spacy_analysis if str(token) not in Preprocessor.ALL_EMOJIS]



	@staticmethod
	def replace_emojis(word:str)->str:
		"""
		Replaces positive emojis with "<postive>" and negative ones with "<negative"
		:param word: input, e.g. "🥰" or "trump"
		:return: emoji converted, everyting else untouched -> "<postive>", "trump"
		"""
		if word in Preprocessor.POSITIVE_EMOJIS:
			word = "<positiveemoji>"
		elif word in Preprocessor.NEGATIVE_EMOJIS:
			word = "<negativeemoji>"
		return word



	@staticmethod
	def reduce_repeated_letters(word:str, target_number=1) -> str:
		"""
		Replace substrings consisting of multiple repetitions of the same character "hellooooo" with just target_number
		instances of that character; so if target_number is 1, we would get "hello", if it is 2, "helloo"
		:param word: word for which the repeated letters should be reduced
		:param target_number: how many repeated letters shall remain after reduction
		:return: reduced word
		"""
		last_character = word[0]
		strings_to_replace = []
		character_string = str(last_character)
		for char in word[1:]:
			if char == last_character:
				character_string += char
			else:
				if len(character_string) > 2:
					strings_to_replace.append(character_string)
				last_character = char
				character_string = str(last_character)
		for charstring in strings_to_replace:
			word = word.replace(charstring, charstring[:target_number])
		return word



	@staticmethod
	def lemmatize(word:str)->str:
		"""
		Lemmatize the input word and return the lemmatized version
		"""
		lem = WordNetLemmatizer()
		return lem.lemmatize(word, "v")


	@staticmethod
	def sentence_remove_stopwords(sentence:str, stopwords:list) ->str:
		punctuation = '[,/.?!:;]'
		for word in sentence:
			if word in stopwords or word[:-1] in stopwords and word[-1] in punctuation:
				sentence.remove(word)
		return sentence

	@staticmethod
	def corpus_remove_stopwords(corpus:list, stopwords:list):
		sentences = []
		for sentence in corpus:
			sentences.append(Preprocessor.sentence_remove_stopwords(sentence, stopwords))
		return sentences



	def preprocess(self,
				   spellcorrect=False,
				   lemmatize=False,
				   reduce_chars=True,
				   emoji_handling=True,
				   verbose_emoji=False,
				   ):
		"""
		Applies the different preprocessing methods
		:param spellcorrect: Should spelling be corrected?
		:param lemmatize: Should words be lemmatized?
		:return: nothing
		"""
		if reduce_chars:
			self.reduce_repeated_letters_corpus()
		if emoji_handling:
			self.emoji_replacement(verbose=verbose_emoji)
		if lemmatize:
			self.lemmatization()
		if spellcorrect:
			self.spelling_correction()


class UniversalPreprocessor(Preprocessor):
		def __init__(self, corpus, tokenize=True):
			super().__init__(corpus=corpus,
							 tokenize=tokenize,
							 url_removal=True,
							 tag_removal=True,
							 emoji_handling=False,
							 verbose_emoji = False,
							 remove_short_tweets=False,
							 spellcorrect=False,
							 lemmatize=False)


class SentimentPreprocessor(Preprocessor):
	def __init__(self, corpus, remove_short_tweets=True, spellcorrect=False, lemmatize=False, tokenize=False):
		super().__init__(corpus=corpus,
						 tokenize=tokenize,
						 url_removal=True,
						 tag_removal=True,
						 emoji_handling=True,
						 verbose_emoji=False,
						 remove_short_tweets=remove_short_tweets,
						 spellcorrect=spellcorrect,
						 lemmatize=lemmatize,
						 stopwords=[])

class LDAPreprocessor(Preprocessor):
	def __init__(self, corpus, stopwords=[], tokenize=True):
		super().__init__(corpus=corpus,
						 emoji_handling=True,
						 verbose_emoji=True,
						 tokenize=tokenize,
						 remove_short_words=True,
						 lemmatize=True,
						 stopwords=stopwords)



class SummarisationPreprocessor(Preprocessor):
	def __init__(self, corpus, spellcorrect=False, tokenize=True):
		super().__init__(corpus=corpus,
						 tokenize=True,
						 remove_short_words=False,
						 lemmatize=False,
						 stopwords=[],
						 remove_short_tweets=True,
						 emoji_handling=False,
						 tag_removal=True,
						 url_removal=True,
						 reduce_chars=True,
						 keep_only_longest_sentence=True,
						 spellcorrect=spellcorrect)
		self.remove_tweets_beginning_with_stopwords()

	def remove_tweets_beginning_with_stopwords(self, stopwords=["and", "or"]):
			for i, tweet in enumerate(self.corpus):
				idx = self.idx[i]
				if tweet[0].lower() in stopwords:
					self.corpus.remove(tweet)
					self.idx.remove(idx)


# if __name__ == "__main__":
# 	tweets = [
# 			"This is a short sentence! 🤠 This is a much much much longer sentence.",
# 			"⚡️ I #met a @trevellller frm an aaaaancent land https://gitlab.doc.ic.ac.uk/sww116/msc-ai-group-project/blob/sentiment-analysis/sentiment-analysis_se/data/sentiment140.csv",
# 			"who said: two vast and trunkless legs of stone 🥰",
# 			"stand in The ☀️ Desssert 😩🦅😊",
# 			"near them, on the sand, a sunk ans scattttered visage lies",
# 			"Biden can't beat Trump!",
# 			"PEDOWOOD!",
# 			"And so on and so on etc...",
# 			"IMPEACHMENT HOAX!",
# 			"Trump sucks! 😡",
# 			"not guilty!",
# 			"not impeachable!"
# 	]
#
# 	prep = LDAPreprocessor(tweets, tokenize=False)
#
# 	print("The preprocessed corpus:")
# 	tweets, indices = prep.preprocessed_corpus()
# 	for x in zip(tweets, indices):
# 		print(x)
# 		print("\n")
#
#
# 	for i, tweet in enumerate(prep.corpus):
# 		emoji_score = prep.get_emoji_score(i)
# 		print(f'The emoji-score of the tweet\n {" ".join(tweet)}\n is: \npositive: {emoji_score["positive"]}\n'
# 			  f'negative: {emoji_score["negative"]}\n')

import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet


## get Antonyms using Wordnet #
def get_antonyms(word):

	antonyms=[]
	for syn in wordnet.synsets(word):
		for l in syn.lemmas():
			if l.antonyms():
				antonyms.append(l.antonyms()[0].name())
	return set(antonyms)



##### Simple Negation with posatag ###


def invert_simple(sent):
	text = word_tokenize(sent)
	tags=pos_tag(text)
	new_sent=[]
	for word in tags:
		if word[0] not in ['not','n\'t']:
			new_sent.append(word[0])
	new_sent=" ".join(new_sent)
	return new_sent
'''
example:
sent="i don't like cooking, it isn't good"
print(invert_simple(sent))	
''''






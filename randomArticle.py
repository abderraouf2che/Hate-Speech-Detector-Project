#This function does the job of extracting a random sentences or articles from Wikipedia
from inscriptis import get_text
import urllib.request

def random_sent(n_sentences=100):
	n=n_sentences
	random_sents=[]
	for i in range(n):

		url = "http://en.wikipedia.org/wiki/Special:Random"
		html = urllib.request.urlopen(url).read().decode('utf-8')

		text = get_text(html)
		text3=text.split('.')
		text3=text3[1]
		random_sents.append(text3)
	return random_sents


text3=random_sent()

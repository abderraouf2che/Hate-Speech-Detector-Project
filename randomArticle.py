#This function does the job of extracting a random sentences or articles from Wikipedia
from inscriptis import get_text
import urllib.request


def random_sent(n_iters=1000):
	'''
	This function generates random sentences from random articles from wikipedia database,
	it stores the dataset in both variable and csv file so when needed.
	'''

    #number of iterations
    n=n_iters

    random_sents=[]
    with open('randomly.csv','w') as file:
        for i in range(n):
		#random article link
            url = "http://en.wikipedia.org/wiki/Special:Random"
            html = urllib.request.urlopen(url).read().decode('utf-8')
            text = get_text(html)

            text=text.split('\n')
		#loop in the text and extract sentences, add them to random_sents and add them to the csv file :
            for i,line in enumerate(text):
                if ' is ' in line:

                    line=text[i]
                    if line[0:8]!='  * Text':

                        random_sents.append(line)
                        file.write((line.encode('ascii','ignore')).decode('utf-8'))
                        file.write('\n')
                    
                    # break

    return random_sents

## write into csv
text=random_sent()
with open('rand.csv','w') as file:
	    for line in text:
	        file.write((line.encode('ascii','ignore')).decode('utf-8'))
	        file.write('\n')

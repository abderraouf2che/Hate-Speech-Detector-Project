#This function does the job of extracting a random sentences or articles from Wikipedia
from inscriptis import get_text
import urllib.request

def random_sent(n_sentences=2):
    n=n_sentences
    random_sents=[]
    for i in range(n):
        url = "http://en.wikipedia.org/wiki/Special:Random"
        html = urllib.request.urlopen(url).read().decode('utf-8')
        text = get_text(html)

        text=text.split('\n')
    # index=text.index('Jump to navigation Jump to search')
    # print(index) 
        for i,line in enumerate(text):
            if ' is ' in line:
                random_sents.append(text[i])
                break
    return random_sents

## write into csv
text=random_sent()
with open('rand.csv','w') as file:
	    for line in text:
	        file.write((line.encode('ascii','ignore')).decode('utf-8'))
	        file.write('\n')

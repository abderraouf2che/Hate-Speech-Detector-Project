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



from nltk.stem.wordnet import WordNetLemmatizer
from thesaurus import Word
from nltk import word_tokenize, pos_tag

sentence='He told us a very exciting adventure story.'

def sent_negation(sentence):
    
    lem=WordNetLemmatizer()
    tags3=pos_tag(word_tokenize(sentence))

    new_st=[]
    for i,word in enumerate(tags3):

        if word[1]=='VBP':
            if word[0] in ['am','are','have']:
                if tags3[i+1][1]=='JJ':
                    new_st.append(word[0])
                    pass
                else:
                    new_st.append(word[0])
                    new_st.append('not')

            else:
                w=Word(word[0])
                if len(w.antonyms())>0:
                    new_st.append(w.antonyms()[0])
                else:
                    new_st.append('do not')
                    new_st.append(word[0])

        elif word[1]=='JJ':
            w=Word(word[0])
            if len(w.antonyms())>0:
                new_st.append(w.antonyms()[0])
            else:
                new_st.append('not')
                new_st.append(word[0])

        elif word[1]=='VBZ':
            if word[0] in ["'s",'is','has']:
                if tags3[i+1][1]=='JJ':
                    new_st.append(word[0])
                    pass
                else:
                    new_st.append(word[0])
                    new_st.append('not')

            else:
                new_st.append('does not')
                new_st.append(lem.lemmatize(word[0],"v"))
        elif word[1]=='VBD':
            if word[0] in ['was','were']:
                if tags3[i+1][1]=='JJ':
                    new_st.append(word[0])
                    pass
                else:
                    new_st.append(word[0])
                    new_st.append('not')
            else:
                new_st.append('did not')
                new_st.append(lem.lemmatize(word[0],"v"))
        elif word[1]=='MD':
            if tags3[i+1][0] in ['not','n\'t']:
                new_st.append(word[0])

            else:
                new_st.append(word[0])
                new_st.append('not')
        elif word[0] in ['not','n\'t']:
            pass
        else:
            new_st.append(tags3[i][0])
    return " ".join(new_st)






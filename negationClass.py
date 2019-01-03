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
	return sorted(set(antonyms))



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
'''



from nltk.stem.wordnet import WordNetLemmatizer
from thesaurus import Word
from nltk import word_tokenize, pos_tag

sentence='He told us a very exciting adventure story.'

def sent_negation(sentence):
    
    lem=WordNetLemmatizer()
    tags3=pos_tag(word_tokenize(sentence))
    tags3.append((" "," "))
    new_st=[]
    for i,word in enumerate(tags3):
        try:
            if word[1]=='VBP':
                if word[0] in ['am','are','have',"'m","'re","'ve","ve","m",'re','im']:
                    if tags3[i+1][1]=='JJ':
                        new_st.append(word[0])
                        pass
                    elif tags3[i+1][0] in ['not','n\'t']:
                        new_st.append(word[0])
                        
                    else:
                        new_st.append(word[0])
                        new_st.append('not')

                elif word[0] in ['do','did']:
                    if tags3[i+1][0] in ['not','n\'t']:
                        new_st.append(word[0])
                        
                    else:
                        new_st.append(word[0])
                        new_st.append('not')
                else:
                    w=word[0]
                    a=get_antonyms(w)
                    if len(a)>0:
                        new_st.append(a[0])
                    else:
                        new_st.append('do not')
                        new_st.append(word[0])

            elif word[1]=='JJ':
                
                a=get_antonyms(word[0])
                if word[0]=='i':
                    new_st.append(word[0])
                    
                elif len(a)>0:
                    new_st.append(a[0])
                    pass
                else:
                    if tags3[i-1][0]=='the':
                        new_st.append('non')
                        new_st.append(word[0])
                    else:
                        
                        new_st.append('not')
                        new_st.append(word[0])

            elif word[1]=='VBZ':
                if word[0] in ["'s",'is','has','does']:
                    if tags3[i+1][1]=='JJ':
                        new_st.append(word[0])
                        pass
                    elif tags3[i+1][0] in ['not','n\'t']:
                        new_st.append(word[0])
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
        except:
            pass
    return " ".join(new_st)






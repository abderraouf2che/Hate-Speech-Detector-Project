import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from sklearn.linear_model import LogisticRegression
import keras
import urllib
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


print('=========================== Data Preparation =====================')

# download annotated comments and annotations for Attack dataset:

ANNOTATED_COMMENTS_URL = 'https://ndownloader.figshare.com/files/7554634' 
ANNOTATIONS_URL = 'https://ndownloader.figshare.com/files/7554637' 


def download_file(url, fname):
    urllib.request.urlretrieve(url, fname)

                
download_file(ANNOTATED_COMMENTS_URL, 'attack_annotated_comments.tsv')
download_file(ANNOTATIONS_URL, 'attack_annotations.tsv')

# Read data files
comments = pd.read_csv('attack_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations = pd.read_csv('attack_annotations.tsv',  sep = '\t')
#download word embedding vector
wiki_link='https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip'
download_file(wiki_link, 'wiki-news-300d-1M.vec')

# labels a comment as an atack if the majority of annoatators did so
labels = annotations.groupby('rev_id')['attack'].mean() > 0.5
# join labels and comments
comments['attack'] = labels

print('=========================== Data Cleaning =====================')
# clean the text
comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.lower())
comments['comment'] = comments['comment'].apply((lambda x: re.sub('.,[^a-zA-z0-9\s]','',x)))
comments['comment'] = comments['comment'].apply((lambda x: re.sub('[/(){}\[\]\|@,;]','',x)))
comments['comment'] = comments['comment'].apply((lambda x: re.sub(' +',' ',x)))
comments['comment'] = comments['comment'].apply((lambda x: re.sub(':',' ',x)))
comments['comment'] = comments['comment'].apply((lambda x: re.sub('`',' ',x)))
comments['comment'] = comments['comment'].apply((lambda x: re.sub('>',' ',x)))
comments['comment'] = comments['comment'].apply((lambda x: re.sub('<',' ',x)))

# keeping only training and test sets
train_comments = comments.query("split=='train'")
valid_comments = comments.query("split=='test'")

# split the dataset into training and validation datasets 
train_x, valid_x = train_comments['comment'], valid_comments['comment'], 
train_y, valid_y = train_comments['attack'], valid_comments['attack']
# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


print('===========================  Creating Vectorizers for data =====================')

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

##### Word Embedding Matrix #####

# load the pre-trained word-embedding vectors 
embeddings_index = {}
for i, line in enumerate(open('data/wiki-news-300d-1M.vec', encoding="utf8")):
    values = line.split()
    embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(comments['comment'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

# create token-embedding mapping
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
print('===========================  Building Pipelines for Data Classification and Prediction =====================')

print('===========================   Naive Bayes =====================')

######
# for Count vectors
clf = Pipeline([
    ('vect',count_vect),
    ('clf', naive_bayes.MultinomialNB()),
])
#fit the model
clf = clf.fit(train_comments['comment'], train_comments['attack'])
# show accuracy Measure
auc = roc_auc_score(valid_comments['attack'], clf.predict_proba(valid_comments['comment'])[:, 1])
print('Test ROC AUC for Naive Bayes with word count Features: %.3f' %auc)
#testing a sentence:
clf.predict(['No, I hate my life, Pierre muttered, wincing'])

######
### Naive Bayes for TFidf word level
clf = Pipeline([
    ('vect',tfidf_vect),
    ('clf', naive_bayes.MultinomialNB()),
])
#fit the model
clf = clf.fit(train_comments['comment'], train_comments['attack'])

# show accuracy Measure
auc = roc_auc_score(valid_comments['attack'], clf.predict_proba(valid_comments['comment'])[:, 1])
print('Test ROC AUC for Naive Bayes with Tfidf word level features: %.3f' %auc)

#testing a sentence:
clf.predict(['No, I hate my life, Pierre muttered, wincing'])

######
### Naive Bayes for TFidf ngram
clf = Pipeline([
    ('vect',tfidf_vect_ngram),
    ('clf', naive_bayes.MultinomialNB()),
])
#fit the model
clf = clf.fit(train_comments['comment'], train_comments['attack'])

# show accuracy Measure
auc = roc_auc_score(valid_comments['attack'], clf.predict_proba(valid_comments['comment'])[:, 1])
print('Test ROC AUC for Naive Bayes with Tfidf ngram features : %.3f' %auc)

#testing a sentence:
clf.predict(['No, I hate my life, Pierre muttered, wincing'])

### Naive Bayes for TFidf ngram character level

clf = Pipeline([
    ('vect',tfidf_vect_ngram_chars),
    ('clf', naive_bayes.MultinomialNB()),
])
#fit the model
clf = clf.fit(train_comments['comment'], train_comments['attack'])

# show accuracy Measure
auc = roc_auc_score(valid_comments['attack'], clf.predict_proba(valid_comments['comment'])[:, 1])
print('Test ROC AUC Naive Bayes for TFidf ngram character level:  %.3f' %auc)

#testing a sentence:
clf.predict(['I hate my life, Pierre muttered, wincing'])


print('===========================   Linear Classifier =====================')


# Linear Classifier Pipeline

########
#on Count Vectors

clf = Pipeline([
    ('vect',count_vect),
    ('clf', linear_model.LogisticRegression()),
])

#fit the model
clf = clf.fit(train_comments['comment'], train_comments['attack'])

# show accuracy Measure
auc = roc_auc_score(valid_comments['attack'], clf.predict_proba(valid_comments['comment'])[:, 1])
print('Test ROC AUC Linear Classifier on count vectors: %.3f' %auc)

#testing a sentence:
clf.predict(['No, I hate my life, Pierre muttered, wincing'])


#########
# Linear Classifier Pipeline
#on TFidf

clf = Pipeline([
    ('vect',tfidf_vect),
    ('clf', linear_model.LogisticRegression()),
])

#fit the model
clf = clf.fit(train_comments['comment'], train_comments['attack'])

# show accuracy Measure
auc = roc_auc_score(valid_comments['attack'], clf.predict_proba(valid_comments['comment'])[:, 1])
print('Test ROC AUC for Linear Classifier Pipeline on TFidf: %.3f' %auc)

#testing a sentence:
clf.predict(['No, I hate my life, Pierre muttered, wincing'])

######
# Linear Classifier Pipeline
#on TFidf ngram

clf = Pipeline([
    ('vect',tfidf_vect_ngram),
    ('clf', linear_model.LogisticRegression()),
])

#fit the model
clf = clf.fit(train_comments['comment'], train_comments['attack'])

# show accuracy Measure
auc = roc_auc_score(valid_comments['attack'], clf.predict_proba(valid_comments['comment'])[:, 1])
print('Test ROC AUC Linear Classifier Pipeline on TFidf ngram: %.3f' %auc)

#testing a sentence:
clf.predict(['No, I hate my life, Pierre muttered, wincing'])

#####
# Linear Classifier Pipeline
#on TFidf

clf = Pipeline([
    ('vect',tfidf_vect_ngram_chars),
    ('clf', linear_model.LogisticRegression()),
])

#fit the model
clf = clf.fit(train_comments['comment'], train_comments['attack'])

# show accuracy Measure
auc = roc_auc_score(valid_comments['attack'], clf.predict_proba(valid_comments['comment'])[:, 1])
print('Test ROC AUC Linear Classifier Pipeline on TFidf ngram character level: %.3f' %auc)

#testing a sentence:
clf.predict(['No, I hate my life, Pierre muttered, wincing'])

####












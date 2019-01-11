#importing dependencies
import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import keras
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from Data_prep import *

# Read data files
comments_attack = pd.read_csv('attack_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations_attack = pd.read_csv('attack_annotations.tsv',  sep = '\t')
comments_aggression = pd.read_csv('aggression_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations_aggression = pd.read_csv('aggression_annotations.tsv',  sep = '\t')
comments_toxicity = pd.read_csv('toxicity_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations_toxicity = pd.read_csv('toxicity_annotations.tsv',  sep = '\t')

########## Data preparation and cleaning #########
comments_attack=prep_comments(comments_attack,annotations_attack, 'attack')
comments_toxicity=prep_comments(comments_toxicity,annotations_toxicity, 'toxicity')
comments_aggression=prep_comments(comments_aggression,annotations_aggression, 'aggression')

# Take neutral, Attack, Aggression, Toxicity
neutral01=comments_attack.query("label == False")
neutral02=comments_aggression.query("label == False")
neutral03=comments_toxicity.query("label == False")
neutral=pd.concat([neutral01, neutral02, neutral03], axis = 0)
comments_attack = comments_attack.query("label == True")
comments_aggression = comments_aggression.query("label == True")
comments_toxicity = comments_toxicity.query("label == True")

# labels: Neutral=0, Attack = 1, Aggression =2, Toxicity = 3
neutral['label']=0
comments_attack['label'] = comments_attack['label']=1
comments_aggression['label'] = comments_aggression['label']=2
comments_toxicity['label'] = comments_toxicity['label']=3

# Concatenation of the four data sets
dataframe = pd.concat([neutral, comments_attack, comments_aggression, comments_toxicity], axis = 0)


#split the data into training and validation sets
train_x, valid_x, train_y, valid_y = train_test_split(dataframe['comment'], dataframe['label'], test_size=0.2, random_state=42)


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


vectorizers=[(count_vect,'count_vectorizer'),(tfidf_vect,'tfidf_vectorizer_word'),(tfidf_vect_ngram,'tfidf_vectorizer_ngram'),(tfidf_vect_ngram_chars,'tfidf_vectorizer_ngram_chars')]

######
#Naive Bayes for all features:
for vectorizer in vectorizers:
  print(vectorizer[1])
  clf = Pipeline([
    ('vect',vectorizer[0]),
    ('clf', naive_bayes.MultinomialNB()),
  ])
  clf = clf.fit(train_comments['comment'], train_comments['aggression'])
  # show accuracy Measure
  auc = roc_auc_score(valid_comments['aggression'], clf.predict_proba(valid_comments['comment'])[:, 1])
  print('Test ROC AUC for '+vectorizer[1]+': %.3f' %auc)
  #testing a sentence:
  print("test for sentence :  == whoa == you are a big fat idot stop spamming my userspace")
  print(clf.predict([' == whoa == you are a big fat idot stop spamming my userspace']))
  print('\n')
  
  print('===========================   Linear Classifier =====================')


# Linear Classifier Pipeline

#Linear Classifier for all features:
for vectorizer in vectorizers:
  print(vectorizer[1])
  clf = Pipeline([
    ('vect',vectorizer[0]),
    ('clf', linear_model.LogisticRegression()),
  ])
  clf = clf.fit(train_comments['comment'], train_comments['aggression'])
  # show accuracy Measure
  auc = roc_auc_score(valid_comments['aggression'], clf.predict_proba(valid_comments['comment'])[:, 1])
  
  print('Test ROC AUC for '+vectorizer[1]+': %.3f' %auc)
  #testing a sentence:
  print("test for sentence : == whoa == you are a big fat idot stop spamming my userspace")
  print(clf.predict(['== whoa == you are a big fat idot stop spamming my userspace']))
  print('\n')
  
  
  
print('========================= . SVM  Classifier ==================')

#### SVM on count vectors:
# SVM Classifier Pipeline
#on word count vector

clf = Pipeline([
    ('vect',count_vect),
    ('clf', svm.SVC(gamma='scale',probability=True)
),
])
#SVM SVC Classifier for all features:
for vectorizer in vectorizers:
  print(vectorizer[1])
  clf = Pipeline([
    ('vect',vectorizer[0]),
    ('clf', svm.SVC(gamma='scale',probability=True)),
  ])
  clf = clf.fit(train_comments['comment'], train_comments['aggression'])
  # show accuracy Measure
  auc = roc_auc_score(valid_comments['aggression'], clf.predict_proba(valid_comments['comment'])[:, 1])
  
  print('Test ROC AUC for '+vectorizer[1]+': %.3f' %auc)
  #testing a sentence:
  print("test for sentence : == whoa == you are a big fat idot stop spamming my userspace")
  print(clf.predict(['== whoa == you are a big fat idot stop spamming my userspace']))
  print('\n')

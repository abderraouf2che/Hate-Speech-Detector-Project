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

################    NEURAL NETWORKS ################
# Feed forward NN with 1 hidden layer
def model_FF(xtrain, ytrain, xvalid, yvalid, hidden_size, epochs =1):
    # create input layer 
    input_layer = layers.Input((xtrain.shape[1], ), sparse=True)
    
    # create hidden layer
    hidden_layer = layers.Dense(hidden_size, activation="relu")(input_layer)
    
    # create output layer
    output_layer = layers.Dense(4, activation="softmax")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    classifier.fit(xtrain, ytrain,
                  batch_size=256,
                  epochs=epochs,
                  shuffle = True)
    # scores of the classifier
    predictions = classifier.predict(xvalid)
    predictions = predictions.argmax(axis=-1)
    accuracy = classifier.evaluate(xvalid, yvalid, verbose=0)
    f1score = metrics.f1_score(valid_y, predictions, average='weighted')
    return accuracy, f1score

# convert to one_hot
train_y_onehot = keras.utils.to_categorical(train_y, 4)
valid_y_onehot = keras.utils.to_categorical(valid_y, 4)


# NN Classifier on Count Vectors
accuracy, f1score = model_FF(xtrain_count, train_y_onehot, xvalid_count, valid_y_onehot, 100)
print("NN, Count Vectors accuracy:%s     f1 score: %s"% (accuracy[1], f1score))

# NN Classifier on Word Level TF IDF Vectors
accuracy, f1score = model_FF(xtrain_tfidf, train_y_onehot, xvalid_tfidf, valid_y_onehot, 100)
print("NN, Count Vectors accuracy:%s     f1 score: %s"% (accuracy[1], f1score))

# NN Classifier on Ngram Level TF IDF Vectors
accuracy, f1score = model_FF(xtrain_tfidf_ngram, train_y_onehot, xvalid_tfidf_ngram, valid_y_onehot, 100)
print("NN, Count Vectors accuracy:%s     f1 score: %s"% (accuracy[1], f1score))

# NN Classifier on Count Vectors
accuracy, f1score = model_FF(xtrain_tfidf_ngram_chars, train_y_onehot, xvalid_tfidf_ngram_chars, valid_y_onehot, 100)
print("NN, Count Vectors accuracy:%s     f1 score: %s"% (accuracy[1], f1score))

############## Convolution Neural Networks

def cnn(xtrain, ytrain, xvalid, yvalid, epochs = 3):
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 4, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(4, activation="softmax")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    model.fit(xtrain, ytrain,
              batch_size=256,
              epochs=epochs)
    predictions = model.predict(xvalid)
    predictions = predictions.argmax(axis=-1)
    accuracy = model.evaluate(xvalid, yvalid, verbose=0)
    f1score = metrics.f1_score(valid_y, predictions, average='weighted')
    return accuracy, f1score
  
accuracy, f1score = cnn(train_seq_x, train_y_onehot, valid_seq_x, valid_y_onehot)
print("CNN, Word Embeddings acuuracy accuracy:%s     f1 score: %s"% (accuracy[1], f1score))

########### LSTM #########
def lstm(xtrain, ytrain, xvalid, yvalid, epochs = 1):
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer1 = layers.LSTM(128)(embedding_layer)
    dropout1 = layers.Dropout(0.5)(lstm_layer1)
    #lstm_layer2 = layers.LSTM(128)(dropout1)
    #dropout2 = layers.Dropout(0.5)(lstm_layer2)
    # Add the output Layers
    output_layer = layers.Dense(4, activation="softmax")(dropout1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(xtrain, ytrain,
              batch_size=256,
              epochs=3)
    
    predictions = model.predict(xvalid)
    predictions = predictions.argmax(axis=-1)
    accuracy = model.evaluate(xvalid, yvalid, verbose=0)
    f1score = metrics.f1_score(valid_y, predictions, average='weighted')
    return accuracy, f1score
    
accuracy, f1score = lstm(train_seq_x, train_y_onehot, valid_seq_x, valid_y_onehot)
print("LSTM, Word Embeddings accuracy:%s     f1 score: %s"% (accuracy[1], f1score))

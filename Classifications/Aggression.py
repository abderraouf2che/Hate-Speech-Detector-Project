#importing dependencies
import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import keras
import re
# Read data files
comments = pd.read_csv('aggression_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations = pd.read_csv('aggression_annotations.tsv',  sep = '\t')

# labels a comment as an atack if the majority of annoatators did so
labels = annotations.groupby('rev_id')['aggression'].mean() > 0.5

# join labels and comments
comments['aggression'] = labels

# clean the text
comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.lower())
comments['comment'] = comments['comment'].apply((lambda x: re.sub('.,[^a-zA-z0-9\s]','',x)))
# keeping only training and test sets
train_comments = comments.query("split=='train'")
valid_comments = comments.query("split=='test'")

# split the dataset into training and validation datasets 
train_x, valid_x = train_comments['comment'], valid_comments['comment'], 
train_y, valid_y = train_comments['aggression'], valid_comments['aggression']
# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(comments['comment'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(comments['comment'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(comments['comment'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(comments['comment'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 

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

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False, epochs=None):
    
    if is_neural_net:
        classifier.fit(feature_vector_train, label ,epochs=epochs)
        predictions = classifier.predict(feature_vector_valid)
        predictions = predictions.argmax(axis=-1)
    else:
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)
        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)
        
    accuracy = metrics.accuracy_score(predictions, valid_y)
    f1score = metrics.f1_score(valid_y, predictions)
    return accuracy, f1score
"""
# Naive Bayes on Count Vectors
accuracy, f1score = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print("NB, Count Vectors:   accuracy: %s      f1 score: %s"% (accuracy,f1score))

# Naive Bayes on Word Level TF IDF Vectors
accuracy, f1score = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("NB, WordLevel TF-IDF:   accuracy: %s     f1 score: %s"% (accuracy,f1score))

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy, f1score = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("NB, N-Gram Vectors:   accuracy: %s     f1 score: %s"% (accuracy,f1score))

# Naive Bayes on Character Level TF IDF Vectors
accuracy, f1score = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("NB, CharLevel Vectors:   accuracy: %s   f1 score: %s"% (accuracy,f1score))
print("===============================================================================")

# Linear Classifier on Count Vectors
accuracy, f1score = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print("LR, Count Vectors:   accuracy: %s   f1 score: %s"% (accuracy,f1score))

# Linear Classifier on Word Level TF IDF Vectors
accuracy, f1score = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print("LR, WordLevel TF-IDF:   accuracy: %s   f1 score: %s"% (accuracy,f1score))

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy, f1score = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("LR, N-Gram Vectors:   accuracy: %s   f1 score: %s"% (accuracy,f1score))

# Linear Classifier on Character Level TF IDF Vectors
accuracy, f1score = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("LR, CharLevel Vectors:   accuracy: %s   f1 score: %s"% (accuracy,f1score))
print("===============================================================================")

# SVM Classifier on Count Vectors
accuracy, f1score = train_model(svm.SVC(gamma='scale'), xtrain_count, train_y, xvalid_count)
print("SVM, Count Vectors:   accuracy: %s   f1 score: %s"% (accuracy,f1score))

# SVM Classifier on Word Level TF IDF Vectors
accuracy, f1score = train_model(svm.SVC(gamma='scale'), xtrain_tfidf, train_y, xvalid_tfidf)
print("SVM, WordLevel TF-IDF:   accuracy: %s   f1 score: %s"% (accuracy,f1score))

# SVM on Ngram Level TF IDF Vectors
accuracy, f1score = train_model(svm.SVC(gamma='scale'), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("SVM, N-Gram Vectors TF-IDF:   accuracy: %s   f1 score: %s"% (accuracy,f1score))

# SVM Classifier on Character Level TF IDF Vectors
#accuracy, f1score = train_model(svm.SVC(gamma='scale'), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
#print("SVM, CharLevel Vectors:   accuracy: %s   f1 score: %s"% (accuracy,f1score))
print("===============================================================================")

# Feed forward NN with 1 hidden layer
def create_model_architecture(input_size, hidden_size):
    # create input layer 
    input_layer = layers.Input((input_size, ), sparse=True)
    
    # create hidden layer
    hidden_layer = layers.Dense(hidden_size, activation="relu")(input_layer)
    
    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer='adam', loss='binary_crossentropy')
    return classifier 

# NN Classifier on Count Vectors
classifier = create_model_architecture(xtrain_count.shape[1], 100)
accuracy, f1score = train_model(classifier, xtrain_count, train_y, xvalid_count, is_neural_net=True, epochs =3)
print("NN, Count Vectors:   accuracy: %s   f1 score: %s"% (accuracy,f1score))

# NN Classifier on Word Level TF IDF Vectors
classifier = create_model_architecture(xtrain_tfidf.shape[1], 100)
accuracy, f1score = train_model(classifier, xtrain_tfidf, train_y, xvalid_tfidf, is_neural_net=True, epochs =3)
print("NN, WordLevel TF-IDF vector:   accuracy: %s   f1 score: %s"% (accuracy,f1score))

# NN Classifier on Ngram Level TF IDF Vectors
classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1], 100)
accuracy = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=True, epochs =3)
print("NN, Ngram Level TF IDF Vectors:   accuracy: %s   f1 score: %s"% (accuracy,f1score))

# NN Classifier on Character Level TF IDF Vectors
#classifier = create_model_architecture(xtrain_tfidf_ngram_chars.shape[1], 100)
#accuracy,f1score = train_model(classifier, xtrain_tfidf_ngram_chars, train_y, xtrain_tfidf_ngram_chars, is_neural_net=True, epochs =3)
#print("NN, CharLevel Vectors:  accuracy: %s   f1 score: %s"% (accuracy,f1score))
"""
print("===============================================================================")
# CNN model
def cnn(train_x, train_y, valid_x, batch_size=128, epochs = 1):
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=epochs)
    
    predictions = model.predict(valid_x)
    predictions = predictions.argmax(axis=-1)
    
    accuracy = metrics.accuracy_score(predictions, valid_y)
    f1score = metrics.f1_score(valid_y, predictions) 
    return accuracy, f1score 

accuracy, f1score = cnn(train_seq_x, train_y, valid_seq_x, 10)
print("CNN, Word Embeddings:   acuuracy: %s   f1 score: %s"% (accuracy,f1score))
print("===============================================================================")
"""
# LSTM model
def lstm(train_x, train_y, valid_x, batch_size=1024, epochs = 10):
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.Bidirectional(layers.LSTM(100))(embedding_layer)
    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=epochs)
    
    predictions = model.predict(valid_x)
    predictions = predictions.argmax(axis=-1)
    
    accuracy = metrics.accuracy_score(predictions, valid_y)
    f1score = metrics.f1_score(valid_y, predictions) 
    return accuracy, f1score
accuracy, f1score = lstm(train_seq_x, train_y, valid_seq_x)
print("LSTM, Word Embeddings:  accuracy: %s   f1 score: %s"% (accuracy,f1score))
"""

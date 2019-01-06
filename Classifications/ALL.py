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

# Read data files
comments_attack = pd.read_csv('attack_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations_attack = pd.read_csv('attack_annotations.tsv',  sep = '\t')
comments_aggression = pd.read_csv('aggression_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations_aggression = pd.read_csv('aggression_annotations.tsv',  sep = '\t')
comments_toxicity = pd.read_csv('toxicity_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations_toxicity = pd.read_csv('toxicity_annotations.tsv',  sep = '\t')

# labels a comment if the majority of annoatators did so
labels_attack = annotations_attack.groupby('rev_id')['attack'].mean() > 0.5
labels_aggression = annotations_aggression.groupby('rev_id')['aggression'].mean() > 0.5
labels_toxicity = annotations_toxicity.groupby('rev_id')['toxicity'].mean() > 0.5

# join labels and comments
comments_attack['label'] = labels_attack
comments_aggression['label'] = labels_aggression
comments_toxicity['label'] = labels_toxicity

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

# Text preprocessing
dataframe['comment'] = dataframe['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
dataframe['comment'] = dataframe['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
dataframe['comment'] = dataframe['comment'].apply(lambda x: x.lower())
dataframe['comment'] = dataframe['comment'].apply((lambda x: re.sub('[/(){}\[\]\|@,;]','',x)))
dataframe['comment'] = dataframe['comment'].apply((lambda x: re.sub('[^0-9a-z #+_]',' ',x)))
dataframe['comment'] = dataframe['comment'].apply((lambda x: re.sub(' +',' ',x)))

#split the data into training and validation sets
train_x, valid_x, train_y, valid_y = train_test_split(dataframe['comment'], dataframe['label'], test_size=0.2, random_state=42)


# create a count vectorizer object 

count_vect = CountVectorizer(analyzer='word', token_pattern=r'.')
count_vect.fit(dataframe['comment'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(dataframe['comment'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)
# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(3,4), max_features=5000)
tfidf_vect_ngram.fit(dataframe['comment'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(dataframe['comment'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 

# load the pre-trained word-embedding vectors 
embeddings_index = {}
for i, line in enumerate(open('data/wiki-news-300d-1M.vec', encoding="utf8")):
    values = line.split()
    embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(dataframe['comment'])
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

# Define the training function
def train_model(classifier, xtrain, ytrain, xvalid, yvalid):
    # fit the training dataset on the classifier
    classifier.fit(xtrain, ytrain)
    # predict the labels on validation dataset
    predictions = classifier.predict(xvalid)
        
    accuracy = metrics.accuracy_score(predictions, yvalid)
    f1score = metrics.f1_score(yvalid, predictions, average='weighted')
    return accuracy, f1score

###############  Naive Bayes

# Naive Bayes on Count Vectors
accuracy, f1score = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count, valid_y)
print("NB, Count Vectors:   accuracy: %s      f1 score: %s"% (accuracy,f1score))
# Naive Bayes on Word Level TF IDF Vectors
accuracy, f1score = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print("NB, WordLevel TF-IDF:   accuracy: %s     f1 score: %s"% (accuracy,f1score))

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy, f1score = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
print("NB, N-Gram Vectors:   accuracy: %s     f1 score: %s"% (accuracy,f1score))

# Naive Bayes on Character Level TF IDF Vectors
accuracy, f1score = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars, valid_y)
print("NB, CharLevel Vectors:   accuracy: %s   f1 score: %s"% (accuracy,f1score))

###############  Linear Classifier

# Linear Classifier on Count Vectors
accuracy, f1score = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count, valid_y)
print("LR, Count Vectors:   accuracy: %s   f1 score: %s"% (accuracy,f1score))

# Linear Classifier on Word Level TF IDF Vectors
accuracy, f1score = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print("LR, WordLevel TF-IDF:   accuracy: %s   f1 score: %s"% (accuracy,f1score))

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy, f1score = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
print("LR, N-Gram Vectors:   accuracy: %s   f1 score: %s"% (accuracy,f1score))

# Linear Classifier on Character Level TF IDF Vectors
accuracy, f1score = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars, valid_y)
print("LR, CharLevel Vectors:   accuracy: %s   f1 score: %s"% (accuracy,f1score))

############# SVM Model
# warning: this one takes long time to process
# SVM Classifier on Count Vectors
accuracy, f1score = train_model(svm.SVC(), xtrain_count, train_y, xvalid_count,valid_y)
print("SVM, Count Vectors:   accuracy: %s   f1 score: %s"% (accuracy,f1score))

# SVM Classifier on Word Level TF IDF Vectors
accuracy, f1score = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print("SVM, WordLevel TF-IDF:   accuracy: %s   f1 score: %s"% (accuracy,f1score))

# SVM on Ngram Level TF IDF Vectors
accuracy, f1score = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
print("SVM, N-Gram Vectors TF-IDF:   accuracy: %s   f1 score: %s"% (accuracy,f1score))

# SVM Classifier on Character Level TF IDF Vectors
accuracy, f1score = train_model(svm.SVC(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars, valid_y)
print("SVM, CharLevel Vectors:   accuracy: %s   f1 score: %s"% (accuracy,f1score))


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

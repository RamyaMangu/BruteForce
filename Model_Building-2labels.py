#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import pandas as pd
import numpy as np
import inflect
import re
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from nltk.tokenize import word_tokenize,sent_tokenize
import gensim.downloader as api
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import contractions
import unidecode
import demoji
from sklearn.preprocessing import LabelBinarizer
from emot.emo_unicode import UNICODE_EMOJI, UNICODE_EMOJI_ALIAS, EMOTICONS_EMO
from flashtext import KeywordProcessor
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential,Model
from keras.layers import Dense,Bidirectional,LSTM,Embedding, GlobalMaxPooling1D, GlobalMaxPooling3D
from tensorflow.keras.layers import Dropout
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tweepy
from pandas import Series,DataFrame
import pandas as pd
import pickle


# # Data Import and Visualization

# In[2]:


df = pd.read_csv('https://query.data.world/s/5xzuftmozqteqbqzopbcgxbjzusyi7')


# In[3]:


labels = df['sentiment']
data_vis ={}
for label in labels:
    data_vis[label] = data_vis.get(label, 0) + 1
fig = plt.figure(figsize = (15, 5))
sentiments = list(data_vis.keys())
values = list(data_vis.values())
# creating the bar plot
plt.bar(sentiments, values)
plt.xlabel("Sentiment")
plt.ylabel("Number of tweets")
plt.title("Number of tweets based on labels")
plt.show()


# # Removing the unwanted labels and encoding the remaining labels

# In[4]:


#converting our  above emotions to the labels and dropping the content
#df['sentiment'] = df['sentiment'].replace(labels)
#df['content']= list(map(lambda x: x.lower(), df['processed']))
df.drop(index=df[df['sentiment'] == 'love'].index, inplace=True)
df.drop(index=df[df['sentiment'] == 'surprise'].index, inplace=True)
df.drop(index=df[df['sentiment'] == 'relief'].index, inplace=True)
df.drop(index=df[df['sentiment'] == 'boredom'].index, inplace=True)
df.drop(index=df[df['sentiment'] == 'worry'].index, inplace=True)
df.drop(index=df[df['sentiment'] == 'anger'].index, inplace=True)
df.drop(index=df[df['sentiment'] == 'hate'].index, inplace=True)
df.drop(index=df[df['sentiment'] == 'empty'].index, inplace=True)
df.drop(index=df[df['sentiment'] == 'enthusiasm'].index, inplace=True)
df.drop(index=df[df['sentiment'] == 'fun'].index, inplace=True)
df.drop(index=df[df['sentiment'] == 'neutral'].index, inplace=True)
encoder = LabelBinarizer()
encoder.fit(df.sentiment.unique())
with open("encoder", "wb") as file:
    pickle.dump(encoder, file)


# In[5]:


df.sentiment.unique()


# # Preprocessing the data

# In[6]:


#reomve pattern function, input_text is the text we want to process, the pattern is the pattern we want to remove
def remove_pattern(input_text, pattern):
    r = re.findall(pattern, input_text)
    for i in r:
        input_text = re.sub(i, '', input_text)
    return input_text


# In[7]:


all_emoji_emoticons = {**EMOTICONS_EMO,**UNICODE_EMOJI_ALIAS, **UNICODE_EMOJI_ALIAS}
all_emoji_emoticons = {k:v.replace(":","").replace("_"," ").strip() for k,v in all_emoji_emoticons.items()}
kp_all_emoji_emoticons = KeywordProcessor()
for k,v in all_emoji_emoticons.items():
    kp_all_emoji_emoticons.add_keyword(k, v)


# In[8]:


#preprocess the data
replacers = {'dm': 'direct message',
 'thx': 'thanks',
 'dming': 'direct messaging',
 'dmed': 'direct messaged',
 'plz': 'please',
 'u': 'you',
 'asap': 'as soon as possible',
 '...': '',
 '. . .': '',
 'r': 'are', 
 'tho': 'though', 'wassup' : 'whats up', 'lol': 'laughing out loud', 'layin': 'laying', 'y' :'why', 'bd' : 'birthday',
  'btw': 'by the way', 'ty': 'thank you', 'brb': 'be right back', 'omg': 'oh my god', 'yup' : 'yes', 'yep' : 'yes'}
def preprocess(df):
    #remove the @
    df['processed'] = np.vectorize(remove_pattern)(df['content'], "@[\w]*")
    
    #convert to lower case
    df.loc[:, 'processed'] = df.loc[:, 'processed'].str.lower()
    
    #converting don't -> do not
    removed = []
    for text in df['processed']:
        txt =[]
        for word in text.split():
            txt.append(contractions.fix(word))
        ex_txt = ' '.join(txt)
        removed.append(ex_txt)
    df.drop('processed', axis = 1, inplace = True)
    df['processed'] = removed
    
    #remove any links
    df['processed'] = df['processed'].str.replace('http[^\s]*',"")
    
    #convert emoji to text
    pr = []
    for text in df['processed']:
        pr.append(kp_all_emoji_emoticons.replace_keywords(text))
    df.drop('processed', axis = 1, inplace = True)
    df['processed'] = pr
    
    #remove the #
    df['processed'] = np.vectorize(remove_pattern)(df['processed'], "#")
    
    # converting unicode to ascii
    stemmer = PorterStemmer()
    removed = []
    for text in df['processed']:
        removed.append(unidecode.unidecode(text))
    df.drop('processed', axis = 1, inplace = True)
    df['processed'] = removed
    
    #remove the special characters, numbers, punctuations
    df['processed'] = df['processed'].str.replace("[^a-zA-Z#]", " ",regex=True)
    
    #converting common acronyms to common words
    df['processed'] = df['processed'] .str.replace('[...…]','').str.split().apply(lambda x: ' '.join([replacers.get(e, e) for e in x]))
    
     #remove the words smaller than 2
    #df['processed'] = df['processed'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    
    #remove white spaces
    df['processed'] = df['processed'].str.strip()
    
    return df


# In[9]:


df_x = df
df_y = df['sentiment']
df_x


# In[11]:


x_train, x_rem, y_train, y_rem = train_test_split(df_x, df_y, test_size=0.2)
x_test_t, x_valid_t, y_test, y_valid = train_test_split(x_rem, y_rem, test_size=0.5)
df_processed_x = preprocess(x_train)
#convert the labels into categroical data using the keras library
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)
y_valid = encoder.transform(y_valid)
#The output of the y_train and y_test will be a binary matrix and has columns equal to the number of categories in the data.
df_processed_x
y_test


# In[12]:


df_processed_x
x_train = df_processed_x['processed']
x_test = x_test_t['content']
x_valid = x_valid_t['content']


# In[13]:


df_processed_x['processed'].iloc[[190]]


# In[14]:


x_train.shape


# In[15]:


x_test.shape


# In[16]:


x_valid.shape


# In[17]:


x_train


# # Tokenizing the data and padding the text

# In[18]:


tokenizer = Tokenizer(lower=True)
combi = pd.concat([x_train, x_test], axis = 0)
tokenizer.fit_on_texts(combi)
#Since the BiLSTM model only considers numeric values, we convert our vector of tokens to numeric sequnces, that is each
#token is represented by its frequency in the text
x_seq = tokenizer.texts_to_sequences(x_train)
X_train = pad_sequences(x_seq, maxlen = 34, padding='post')
y_seq = tokenizer.texts_to_sequences(x_test)
X_test = pad_sequences(y_seq, maxlen = 34, padding='post')
z_seq = tokenizer.texts_to_sequences(x_valid)
X_valid = pad_sequences(z_seq, maxlen = 34, padding='post')
word_index = tokenizer.word_index


# # Creating Word Embeddings

# In[19]:


# Glove vector contains a 50 dimensional vector corresponding to each word in dictionary.
vocab = 'glove.6B.50d.txt'
# embeddings_index is a dictionary which contains the mapping of
# word with its corresponding 50d vector.
embeddings_index = {}
with open(vocab, encoding='utf8') as f:
    for line in f:
        # splitting each line of the glove.6B.50d in a list of items- in which
        # the first element is the word to be embedded, and from second
        # to the end of line contains the 50d vector.
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs


# In[20]:


embedding_matrix = np.zeros((len(word_index)+1, 50))
for word, index in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
    else:
        embedding_matrix[index] = np.zeros(50)


# # Building the Baseline Model and Predicting the results

# In[21]:


#EarlyStopping and ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
mc = ModelCheckpoint('./model.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)


# In[22]:


def baseline_model(X,Y, classes):
    model=Sequential()
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(classes, activation = 'softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    ep = model.fit(X, Y, batch_size=64, epochs=10, verbose = 1, validation_data=(X_valid, y_valid))
    return model, ep


# In[23]:


#Building the baseline model with only 3 labels
baselineModel, base_his = baseline_model(X_train,y_train, 2)


# In[25]:


#Predicting with only 3 labels - Baseline Model
plt.plot(base_his.history['accuracy'],c='b',label='train accuracy')
plt.plot(base_his.history['val_accuracy'],c='r',label='validation accuracy')
plt.title("Baseline model")
plt.legend(loc='lower right')
plt.show()


# In[26]:


#Results with only 3 labels - Baseline Model
y_pred = np.argmax(baselineModel.predict(X_test), axis  =  1)
y_true = np.argmax(y_test, axis = 1)
from sklearn import metrics
print(metrics.classification_report(y_pred, y_true))


# # Building the BiLSTM Model and Predicting the results

# In[33]:


def build_model(X,Y, classes):
    #Dimension of our embeddings
    embedding_dim = 50
    model=Sequential()
    model.add(Embedding(input_dim=len(word_index)+1, output_dim=50, input_length=len(X[0]), weights = [embedding_matrix], trainable=False))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(classes, activation = 'softmax'))
    adam = keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    ep = model.fit(X, Y, batch_size=128, epochs=25, verbose = 1, validation_data=(X_valid, y_valid), callbacks= [es, mc])
    return model, ep


# In[34]:


#Building model with only 3 labels
bilstmModel, his=build_model(X_train,y_train, 2)


# In[37]:


#Predicting with only 3 labels - bilstm model
plt.plot(his.history['accuracy'],c='b',label='train accuracy')
plt.plot(his.history['val_accuracy'],c='r',label='validation accuracy')
plt.title("BiLSTM model")
plt.legend(loc='lower right')
plt.show()


# In[38]:


#Results with only 3 labels - bilstm model
y_pred =   np.argmax(bilstmModel.predict(X_valid), axis  =  1)
y_true = np.argmax(y_valid, axis = 1)
from sklearn import metrics
print(metrics.classification_report(y_pred, y_true))


# # Extracting the tweets

# In[39]:


#Put your Bearer Token in the parenthesis below
client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAHiaigEAAAAA1hzpI79DUjAi9q8PvGD7lfTzWjQ%3Dw9PNTdJGBWQOWxJjY9l5yP1Z7fglAF5SQGbLN6LrzUHQ5Gvbkd')

# Get tweets that contain the hashtag #petday
# -is:retweet means I don't want retweets
# lang:en is asking for the tweets to be in english
query = 'covid -is:retweet'
tweets = client.search_recent_tweets(query = query, tweet_fields=['context_annotations', 'created_at'], max_results=100)

tweet_lst = [tweet.text for tweet in tweets.data]



data = {'content': tweet_lst}
tweet_df = DataFrame(data)
tweet_pro = preprocess(tweet_df)
tweet_pro


# In[40]:


x_dat = tweet_pro['processed']
x_dat


# # Predicting the emotion from tweets

# In[41]:


x_seq = tokenizer.texts_to_sequences(x_dat)
x_data = pad_sequences(x_seq, maxlen = 34, padding='post')
y_pred = bilstmModel.predict(x_data)


# In[42]:


#Building model with only 3 labels
with open('encoder', 'rb') as file:
    encoder = pickle.load(file)
for index, value in enumerate(np.sum(y_pred, axis=0) / len(y_pred)):
    print(encoder.classes_[index] + ": " + str(value))


# # Extracting the tweets - Happy

# In[43]:


#Put your Bearer Token in the parenthesis below
client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAHiaigEAAAAA1hzpI79DUjAi9q8PvGD7lfTzWjQ%3Dw9PNTdJGBWQOWxJjY9l5yP1Z7fglAF5SQGbLN6LrzUHQ5Gvbkd')

# Get tweets that contain the hashtag #petday
# -is:retweet means I don't want retweets
# lang:en is asking for the tweets to be in english
query = 'happy -is:retweet'
tweets = client.search_recent_tweets(query = query, tweet_fields=['context_annotations', 'created_at'], max_results=100)

tweet_lst = [tweet.text for tweet in tweets.data]



data = {'content': tweet_lst}
tweet_df = DataFrame(data)
tweet_pro = preprocess(tweet_df)
tweet_pro


# In[44]:


x_dat = tweet_pro['processed']
x_dat


# In[45]:


x_seq = tokenizer.texts_to_sequences(x_dat)
x_data = pad_sequences(x_seq, maxlen = 34, padding='post')
y_pred = bilstmModel.predict(x_data)


# In[46]:


#Building model with only 3 labels
with open('encoder', 'rb') as file:
    encoder = pickle.load(file)
for index, value in enumerate(np.sum(y_pred, axis=0) / len(y_pred)):
    print(encoder.classes_[index] + ": " + str(value))


# #  Extracting the tweets - Sad

# In[47]:


#Put your Bearer Token in the parenthesis below
client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAHiaigEAAAAA1hzpI79DUjAi9q8PvGD7lfTzWjQ%3Dw9PNTdJGBWQOWxJjY9l5yP1Z7fglAF5SQGbLN6LrzUHQ5Gvbkd')

# Get tweets that contain the hashtag #petday
# -is:retweet means I don't want retweets
# lang:en is asking for the tweets to be in english
query = 'sad -is:retweet'
tweets = client.search_recent_tweets(query = query, tweet_fields=['context_annotations', 'created_at'], max_results=100)

tweet_lst = [tweet.text for tweet in tweets.data]



data = {'content': tweet_lst}
tweet_df = DataFrame(data)
tweet_pro = preprocess(tweet_df)
tweet_pro


# In[48]:


x_dat = tweet_pro['processed']
x_dat


# In[49]:


x_seq = tokenizer.texts_to_sequences(x_dat)
x_data = pad_sequences(x_seq, maxlen = 34, padding='post')
y_pred = bilstmModel.predict(x_data)


# In[50]:


#Building model with only 3 labels
with open('encoder', 'rb') as file:
    encoder = pickle.load(file)
for index, value in enumerate(np.sum(y_pred, axis=0) / len(y_pred)):
    print(encoder.classes_[index] + ": " + str(value))


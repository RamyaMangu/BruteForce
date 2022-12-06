External libraries
-
  - nltk.corpus package (https://www.nltk.org/api/nltk.corpus.html)
  - nltk package stopwords (https://www.nltk.org/search.html?q=stopwords)
  - tweepy package (https://www.tweepy.org/)
  - inflect (https://pypi.org/project/inflect/)
  - keras package (https://keras.io/)
    * keras.utils
    * keras.models
    * keras.layers
    * tensorflow.keras.layers
    * tensorflow.compat.v1.keras.layers
    * keras.callbacks
    * keras.preprocessing.text
    * keras_preprocessing.sequence
  - contractions (https://pypi.org/project/contractions/)
  - demoji (https://pypi.org/project/demoji/)
  - emot (https://pypi.org/project/emot/)
  - flashtext (https://pypi.org/project/flashtext/)
  
Publicly available code
-
  - GitHub for version control and coordinating with teammates
  - Jupyter Notebook with Python support for consolidating code
  - Contractions library to expand words
  - UNICODE to remove the emojis (Modified to remove more emojis)
  - NLTK for removing stop words, punctuation, and non-alphabetical characters (Modified to change and remove more words)
  - Keras for building the BiLTSM model (Added to a better model)
  - GloVe 50d embeddings to create the weighted matrix for the embedding layer
  - Sklearn for classification report (i.e. to calculate the Recall, Precision, and F1 scores) (Added to a better model)
  - Tweepy library to scrape tweets from Twitter API (Modified to get different queries)

Our written code
-
  - preprocess.ipynb Collect tweets according to hashtags using the Tweepy library
  - preprocess.ipynb Preprocessing the text
  - Model_Building-3labels.py Splitting data into 80% training, 10% validation and 10% testing data 
  - Model_Building-3labels.py Creating the weighted matrix
  - Model_Building-3labels.py Building the BiLSTM model 
  - Model_Building-3labels.py Extracting the Tweets from the Teepy library

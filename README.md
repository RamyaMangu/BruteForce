# External libraries:

* nltk.corpus package (https://www.nltk.org/api/nltk.corpus.html)
* tweepy package (https://www.tweepy.org/)
* inflect (https://pypi.org/project/inflect/)
* keras package (https://keras.io/)
* keras.utils
* keras.models
* keras.layers
* tensorflow.keras.layers
* tensorflow.compat.v1.keras.layers
* keras.callbacks
* keras.preprocessing.text
* keras_preprocessing.sequence
* contractions (https://pypi.org/project/contractions/)
* demoji (https://pypi.org/project/demoji/)
* emot (https://pypi.org/project/emot/)
* flashtext (https://pypi.org/project/flashtext/)


# Publicly available code:

* GitHub for version control and coordinating with teammates
* Jupyter Notebook with Python support for consolidating code
* Contractions library to expand words
* NLTK for removing  punctuation, and non-alphabetical characters (Modified to change and remove more words)
* Keras for building the BiLTSM model (Added to a better model)
* GloVe 50d embeddings to create the weighted matrix for the embedding layer
* Sklearn for classification report (i.e. to calculate the Recall, Precision, and F1 scores) (Added to a better model)
* Tweepy library to scrape tweets from Twitter API (Modified to get different queries)

# Our written code:

* project.ipynb - application of our model on few hashtags
* preprocess.ipynb-  preprocesses all the data using the libraries mentioned above and tokenizes the data
* WordEmbedds.ipynb-  Created weighted matrix based on the Glove 50d vector
* BaselineModel.ipynb- Creates a baseline model using the keras library
* BiLSTM_Model.ipynb- Building the BiLSTM model
* Application.ipynb- Extracting the Tweets from the Tweepy library and predicting the emotions of the text

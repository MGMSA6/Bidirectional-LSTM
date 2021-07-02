# Fake News Classifier Using Bidirectional LSTM


# Import libraries
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import re
import nltk


# Import dataset
df = pd.read_csv('dataset/train.csv')

# Drop nan values
df = df.dropna()

# Define the Independent and dependent variables
X = df.drop('label', axis=1)
y = df['label']

print(y.value_counts())
print(X.shape, y.shape)

# Vocabulary size
voc_size = 5000

# OneHot Representation

messages = X.copy()

print(messages['title'][1])

messages.reset_index(inplace=True)

nltk.download('stopwords')

# Dataset Preprocessing
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    print(i)
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

print(corpus)
print(len(corpus))
print(voc_size)

# OneHot Representation
onehot_repr = [one_hot(words, voc_size) for words in corpus]

print(onehot_repr)
print(len(onehot_repr))

# Embedding Representation
sent_length = 20
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
print(embedded_docs)

print(embedded_docs[1])

X_final = np.array(embedded_docs)
y_final = np.array(y)

print(X_final.shape, y_final.shape)

# Splitting the data into train and test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

# Define the model
embedding_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size, embedding_vector_features, input_length=sent_length))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# Model Training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)


# Performance Metrics And Accuracy

y_pred = model.predict_classes(X_test)

cm = confusion_matrix(y_pred, y_test)
print(cm)

score = accuracy_score(y_test, y_pred)
print(score)

class_report = classification_report(y_test, y_pred)
print(class_report)

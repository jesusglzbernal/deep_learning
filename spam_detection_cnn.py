import tensorflow as tf
import numpy as np
import pandas as pd
import requests
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import MaxPooling1D, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

url = 'https://lazyprogrammer.me/course_files/spam.csv'
r = requests.get(url)
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
print(df.head())
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
print(df.head())
df.columns = ['labels', 'data']
print(df.head())

# Create binary labels for the CNN
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
y = df['b_labels'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['data'], y, test_size=0.33)

# Convert the sentences into sequences
MAX_VOCAB_SIZE = 20000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(X_train)
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

# Get word -> integer mapping
word2idx = tokenizer.word_index
V = len(word2idx)
print('Found %s unique tokens.' % V)

# Pad the sequences so that we get a N x T matrix
data_train = pad_sequences(sequences_train)
# get the sequence length
T = data_train.shape[1]
data_test = pad_sequences(sequences_test, maxlen=T)
print('Shape of data train tensor:', data_train.shape)
print('Shape of data test tensor:', data_test.shape)

# Create the model

# Set the embedding dimensionality
D = 20

# Note: we actually want the size of the embedding be (V + 1) x D
# because the first index starts from 1 and not 0, (0 is for padding)
# Then, if the last index of the embedding matrix is V,
# then the size of the embedding should be V + 1.

i = Input(shape=(T,))
x = Embedding(V + 1, D)(i)
x = Conv1D(32, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(64, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i, x)

# Compile and Fit the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Training the model')
r = model.fit(data_train,
              y_train,
              epochs=5,
              validation_data=(data_test, y_test))

# Plot the loss per iteration
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Plot the accuracy per iteration
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

# Predict the test set
p_test = model.predict(data_test).argmax(axis=1)
print(X_test)
X_test = X_test.reset_index(drop=True)
print(X_test)
# Show some misclassified examples
misclassified_idx = np.where(p_test != y_test)[0]
print("")
print("Number of errors:", len(misclassified_idx))
print("len X_test:", X_test.shape[0])
print("len p_test:", len(p_test))
print("len y_test:", len(y_test))
for k in range(5):
    i = np.random.choice(misclassified_idx)
    print("     index:", i)
    print("e-mail:", data_test[i])
    print("e-mail text:", X_test[i])
    print("True label: %s Predicted: %s" % (y_test[i],
                                            p_test[i]))
    print("")

print(model.summary())

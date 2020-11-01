import tensorflow as tf
import numpy as np
import itertools
import sys
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, \
                                    Conv2D, \
                                    Dense, \
                                    Dropout, \
                                    BatchNormalization, \
                                    Flatten, \
                                    MaxPooling2D

labels = '''airplane
            automobile
            bird
            cat
            deer
            dog
            frog
            horse
            ship
            truck'''.split()

# Plotting the confusion matrix
def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

print("labels:", labels)

cifar = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar.load_data()
X_train, X_test = X_train/255.0, X_test/255.0
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print(y_train)
y_train, y_test = y_train.flatten(), y_test.flatten()
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print(y_train)
print("X_train[0].shape:", X_train[0].shape)

# Number of distinct classes
k = len(set(y_train))
print("Number of classes:", k)
print("Classes:", set(y_train))

# Defining the model
i = Input(shape=X_train[0].shape)
#x = Conv2D(32, (3, 3), strides=2, padding='same', activation='relu')(i)
#x = Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(x)
#x = Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(k, activation='softmax')(x)

model = Model(i, x)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)

# Plotting the loss
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Plotting the accuracy
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

# Evaluating the model
print(model.evaluate(X_test, y_test))

p_test = model.predict(X_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))

# Show some misclassified eamples
misclassified_idx = np.where(p_test != y_test)[0]
for k in range(5):
    i = np.random.choice(misclassified_idx)
    plt.imshow(X_test[i], cmap='gray')
    plt.title("True label: %s Predicted: %s" % (labels[y_test[i]],
                                                labels[p_test[i]]))
    plt.show()

batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train_generator = data_generator.flow(X_train, y_train, batch_size)
steps_per_epoch = X_train.shape[0]  # The batch size
steps_per_epoch = 1562
r = model.fit_generator(train_generator, validation_data=(X_test, y_test), steps_per_epoch=steps_per_epoch, epochs=1)

# Plotting the loss
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Plotting the accuracy
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

# Evaluating the model
print(model.evaluate(X_test, y_test))

p_test = model.predict(X_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))

# Show some misclassified examples
misclassified_idx = np.where(p_test != y_test)[0]
for k in range(5):
    i = np.random.choice(misclassified_idx)
    plt.imshow(X_test[i], cmap='gray')
    plt.title("True label: %s Predicted: %s" % (labels[y_test[i]],
                                                labels[p_test[i]]))
    plt.show()


print(model.summary())

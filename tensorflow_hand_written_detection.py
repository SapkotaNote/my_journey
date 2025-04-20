import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model = Sequential([
    Flatten(input_shape = (28,28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
train_loos, train_accuracy = model.evaluate(x_test, y_test)
print(f'accuracy: {train_accuracy}')
index = np.random.randint(0, len(x_test))
sample_image = x_test[index]
sample_label = y_test[index]
true_class = np.argmax(sample_label)
predict = model.predict(np.expand_dims(sample_image, axis=0))
predict = np.argmax(predict)

plt.imshow(sample_image, cmap='gray')
plt.title(f"True: {true_class} Predict: {predict}")
plt.axis('off')
plt.show()

print(f"true = {true_class} Predict = {predict} ")
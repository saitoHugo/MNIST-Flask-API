"""
Script que treina e salva (.h5) o modelo no final  
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

#Load the dataset
mnist = tf.keras.datasets.mnist


#Split the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


#Define the model
def create_model():
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model
model = create_model()


#Train the model
model.fit(x_train, y_train, epochs=5)

#Test the model
model.evaluate(x_test,  y_test, verbose=2)

#Save the model
model.save('MNIST.h5')

# Initialize the model
model = tf.keras.models.Sequential()

# Configuring layers
# Input layer using the rectified linear unit activation function
model.add(tf.keras.layers.Dense(11, activation='relu', input_shape=(11,)))

# Additional hidden layers
model.add(tf.keras.layers.Dense(11, activation='relu'))

# Output layer using the sigmoid (s-function) activation function
model.add(tf.keras.layers.Dense(10, activation='sigmoid'))

# Compile and fit the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(X_train, y_train, epochs=200, validation_split=0.1, shuffle=True, callbacks=[tensorboard])

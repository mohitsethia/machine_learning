import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(f"loaded the training data successfully\n")
print(f"the shape of the training data is: ${x_train[0].shape}\n")
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()
plt.imshow(x_train[1], cmap='gray')
plt.title(f"Label: {y_train[1]}")
plt.show()
plt.imshow(x_train[2], cmap='gray')
plt.title(f"Label: {y_train[2]}")
plt.show()
plt.imshow(x_train[3], cmap='gray')
plt.title(f"Label: {y_train[3]}")
plt.show()

# Normalize the data to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape to add a channel dimension (for CNN)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes (digits 0–9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")

# Optional: Predict and show a test image
import numpy as np
i = np.random.randint(0, len(x_test))
prediction = model.predict(x_test[i:i+1])
plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {prediction.argmax()}, Actual: {y_test[i]}")
plt.axis('off')
plt.show()

import tensorflow as tf
import time

# Print device info
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow Version:", tf.__version__)

# Generate synthetic dataset
X = tf.random.normal([10000, 1000])
y = tf.random.normal([10000, 1])

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train and measure time
start = time.time()
history = model.fit(X, y, epochs=5, batch_size=64, verbose=1)
end = time.time()

print("\nTraining Time: {:.2f} seconds".format(end - start))

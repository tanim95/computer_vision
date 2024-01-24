import tensorflow as tf
from tensorflow import keras
from keras import layers

generator = keras.Sequential(
    [
        keras.Input(shape=(100,)),
        layers.Dense(7 * 7 * 256),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, kernel_size=5, strides=1,
                               padding="same", activation="relu"),
        layers.Conv2DTranspose(64, kernel_size=5, strides=2,
                               padding="same", activation="relu"),
        layers.Conv2DTranspose(1, kernel_size=5, strides=2,
                               padding="same", activation="tanh"),
    ],
    name="generator",
)

discriminator = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, kernel_size=5, strides=2, padding="same",
                      activation=keras.layers.LeakyReLU(0.3)),
        layers.Dropout(0.3),
        layers.Conv2D(128, kernel_size=5, strides=2, padding="same",
                      activation=keras.layers.LeakyReLU(0.3)),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)

# Combining the generator and discriminator into a single model
gan = keras.Sequential([generator, discriminator])

# Compile the discriminator
discriminator.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(
    learning_rate=0.0003, beta_1=0.5), metrics=["accuracy"])

# Compile the GAN
gan.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(
    learning_rate=0.0003, beta_1=0.5))

# Loading the MNIST dataset
(x_train, _), (_, _) = keras.datasets.mnist.load_data()

# Normalizing the data
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Training
epochs = 20
batch_size = 128
steps_per_epoch = int(x_train.shape[0] / batch_size)
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for step in range(steps_per_epoch):
        # Generate random noise
        noise = tf.random.normal(shape=(batch_size, 100))
        # Generate fake images
        fake_images = generator(noise)
        # Combine real and fake images
        combined_images = tf.concat(
            [fake_images, x_train[step * batch_size: (step + 1) * batch_size]], axis=0)
        # Labels for real and fake images
        labels = tf.concat(
            [tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0)
        # Add random noise to the labels
        labels += 0.05 * tf.random.uniform(tf.shape(labels))
        # Train the discriminator
        d_loss = discriminator.train_on_batch(combined_images, labels)
        # Generate random noise
        noise = tf.random.normal(shape=(batch_size, 100))
        # Labels for fake images
        misleading_labels = tf.ones((batch_size, 1))
        # Train the generator
        g_loss = gan.train_on_batch(noise, misleading_labels)
    print(f"Discriminator loss: {d_loss}")
    print(f"Generator loss: {g_loss}")


# Function to generate images based on random noise
def generate_images(generator_model, num_images=1):
    noise = tf.random.normal(shape=(num_images, 100))
    generated_images = generator_model.predict(noise)
    return generated_images


# number of images to generate
num_of_img = 5
generated_images = generate_images(generator, num_of_img)

plt.figure(figsize=(15, 3))
for i in range(num_of_img):
    plt.subplot(1, num_of_img, i + 1)
    plt.imshow(generated_images[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.show()

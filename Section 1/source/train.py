# Code inspired by
# https://blog.keras.io/building-autoencoders-in-keras.html
# (also visit this post to learn more about autoencoders)
#
# 0. Import all the necessary elements
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from noisy_digits import get_noisy_mnist
from keras.callbacks import TensorBoard

# 1. Building model.
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

# 2. Compiling model.
model.compile(optimizer='adadelta', loss='binary_crossentropy')

# 3. Getting data.
x_train, x_train_noisy, x_test, x_test_noisy = get_noisy_mnist()

# 4. Training data.
model.fit(x_train_noisy, x_train,
                epochs=2, # to get significant results change to 100
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

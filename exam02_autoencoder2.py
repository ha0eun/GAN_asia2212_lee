import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist

input_img = Input(shape=(784, ))    # 모델성능 좋은 확인하려고 많이줌/ 원본
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='sigmoid')(encoded)  # 위에 encoder dense(64) 받음
decoded = Dense(128, activation='sigmoid')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded) # 가장위 원본 받음
autoencoder = Model(input_img, decoded) # (입력, 출력)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()   # y_train 없으면 라벨이 필요없음/ 자기 지도 학습,
x_train = x_train / 255     # 복합연산자
x_test = x_test / 255

flatted_x_train = x_train.reshape(-1, 784)
flatted_x_test = x_test.reshape(-1, 784)

fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train, epochs = 50,
                           batch_size=256, validation_data=(flatted_x_test, flatted_x_test))

encoded_img = autoencoder.predict(flatted_x_test[:10])

n = 10
plt.gray()
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + n)
    plt.imshow(x_test[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()






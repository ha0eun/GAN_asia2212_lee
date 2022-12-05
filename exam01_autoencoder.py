import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist

input_img = Input(shape=(784, ))    # 이미지가 아니라 쭉 나열해서 줄 것임
encoded = Dense(32, activation='relu')
encoded = encoded(input_img)  # dense 레이어의 출력이면서 인풋을 이미지
decoded = Dense(784, activation='sigmoid') # 출력을 벹어냄/ 입력값을 0과 1사이로 받는 민맥스 정교화
decoded = decoded(encoded)  # 위에 있는 32를 받음
autoencoder = Model(input_img, decoded) #(입력, 출력)
autoencoder.summary()   # 중간 레이어의 출력물을 받고 싶어서 나눠서 작업

encoder = Model(input_img, encoded)
encoder.summary()

encoder_input = Input(shape=(32,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoder_input, decoder_layer(encoder_input))
decoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()   # y_train 없으면 라벨이 필요없음/ 자기 지도 학습,
x_train = x_train / 255     # 복합연산자
x_test = x_test / 255

flatted_x_train = x_train.reshape(-1, 784)
flatted_x_test = x_test.reshape(-1, 784)

fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train, epochs = 50,
                           batch_size=256, validation_data=(flatted_x_test, flatted_x_test))

encoded_img = encoder.predict(x_test[:10].reshape(-1, 784))
decoded_img = decoder.predict(encoded_img)

n = 10
plt.gray()
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()






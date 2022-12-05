import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist


input_img = Input(shape=(28, 28, 1))
x = Conv2D(16,(3, 3), activation='relu', padding='same')(input_img)
x = MaxPool2D((2, 2), padding='same')(x)
x = Conv2D(8,(3, 3), activation='relu', padding='same')(x)
x = MaxPool2D((2, 2), padding='same')(x)
x = Conv2D(8,(3, 3), activation='relu', padding='same')(x)
encoded = MaxPool2D((2, 2), padding='same')(x)  # 4 * 4

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x) # 오버 샘플링 - 8 * 8
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)  # 16 * 16
x = Conv2D(8, (3,3), activation='relu')(x) # 16 -4 = 28
x = UpSampling2D((2, 2))(x)  # 32 * 32
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # 위에 컨브에서 28 맞춰줘서 28 사이즈 크기 맞춤


autoencoder = Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam',loss='binary_crossentropy')

(x_train,_),(x_test,_) = mnist.load_data()   # y_train 없으면 라벨이 필요없음/ 자기 지도 학습,
x_train = x_train /255# 복합연산자
x_test = x_test /255
# print(x_train.shape)
# print(x_train[0])
conv_x_train = x_train.reshape(-1,28, 28, 1)    # 앞에 1주면 1개씩 묶이기 때문에 -1준것은 배수 알아서 찾아가라고 받음
conv_x_test = x_test.reshape(-1, 28, 28 ,1)
# print(conv_x_train.shape)
# print(conv_x_train)

noise_factor = 0.5  # 잡음의 크기
# conv xtrain에 덮어쓰기
conv_x_train_noisy = conv_x_train + np.random.normal(0, 1, size=conv_x_train.shape) * noise_factor  # (평균0, 표준편차1)
conv_x_train_noisy = np.clip(conv_x_train_noisy, 0.0, 1.0) # clip: 0.0이하면 0으로 맞추고 1.0이상되는 것은 1로 덮어서 사용
conv_x_test_noisy = conv_x_test + np.random.normal(0, 1, size=conv_x_test.shape) * noise_factor  # (평균0, 표준편차1)
conv_x_test_noisy = np.clip(conv_x_test_noisy, 0.0, 1.0)
# 잡음이 섞인 데이터


# plt.gray()
plt.figure(figsize=(20, 4))
n =10
for i in range(n):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(conv_x_test_noisy[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i +1+ n)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()




fit_hist = autoencoder.fit(conv_x_train,conv_x_train,epochs =50,
                            batch_size=256,validation_data=(conv_x_test_noisy,conv_x_test))
autoencoder.save('./models/autoencoder_noisy.h5') # 확장자명 중요함!

decoded_img = autoencoder.predict(conv_x_test[:10])


n =10
# plt.gray()
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(conv_x_test_noisy[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i +1+ n)
    plt.imshow(decoded_img[i].reshape(28,28))  # decoded_img 학습시키고 만들어낸 이미지
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()






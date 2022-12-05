import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist
import os

OUT_DIR = './DNN_out'
img_shape = (28, 28, 1)
epochs = 100000
batch_size = 128
noise = 100
sample_interval = 100       # 100마다 샘플뽑아서 저장

(x_train, _), (_, _) = mnist.load_data()     # _ == ytrain, xtest, ytest 필요없어서 _로 표시
print(x_train.shape)

x_train = x_train / 127.5 - 1   # -1 에서 1 사이의 값이 나옴
x_train = np.expand_dims(x_train, axis=3)   # expand_dims : 익스펜드 디멘션 -> 차원을 하나 늘려라
print(x_train.shape)

generator = Sequential()
generator.add(Dense(128, input_dim=noise, activation='leaky_relu'))  # 레이어 100개
generator.add(LeakyReLU(alpha=0.01)) # 진한 부분만 보고 처리할라고 리키렐루에 적용
generator.add(Dense(784, activation='tanh'))  # 하이퍼볼릭 탄젠트: 마이너스 값이라서 tanh을 적용
generator.add(Reshape(img_shape))
generator.summary()


lrelu = LeakyReLU(alpha=0.01) # 디폴트 값 : 0.3

discriminator = Sequential()
discriminator.add(Flatten(input_shape=img_shape))
discriminator.add(Dense(128, activation=lrelu))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.summary()
discriminator.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
discriminator.trainable = False

gan_model = Sequential()
gan_model.add(generator)
gan_model.add(discriminator)
gan_model.summary()
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

real = np.ones((batch_size, 1))     # np.ones : 1로 채워진 행렬로 만들어 주는 것
print(real)
fake = np.zeros((batch_size, 1))
print(fake)


for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0], batch_size)    # 0-59999 랜덤으로 뽑아냄
    real_imgs = x_train[idx]

    z = np.random.normal(0, 1, (batch_size, noise))
    fake_imgs = generator.predict(z)

    d_hist_real = discriminator.train_on_batch(real_imgs, real)    # train_on_batch: 1회만 하고 그만둠
    d_hist_fake = discriminator.train_on_batch(fake_imgs, fake)

    d_loss, d_acc = np.add(d_hist_fake, d_hist_real) * 0.5  # 평균을 구함

    if epoch % 2 == 0:
        z = np.random.normal(0, 1, (batch_size, noise))
        gan_hist = gan_model.train_on_batch(z, real)    # 1이라고 답하게 학습

    if epoch % sample_interval ==0:
        print('%d, [D loss: %f, acc.: %.2f%%], [G loss: %f]'%(
                epoch, d_loss, d_loss, gan_hist))
        row = col = 4
        z = np.random.normal(0, 1 ,(row*col, noise))
        fake_imgs = generator.predict(z)
        fake_imgs = 0.5 * fake_imgs + 0.5

        _, axs = plt.subplots(row, col, figsize=(5, 5),sharey =True, sharex=True)
        cont = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(fake_imgs[cont, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cont += 1
        path = os.path.join(OUT_DIR, 'img-{}'.format(epoch + 1))
        plt.savefig(path)
        plt.close()







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

discriminator = Sequential()        # discriminator : 이진분류기
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
    real_imgs = x_train[idx]    # 랜덤하게 이미지 뽑음

    z = np.random.normal(0, 1, (batch_size, noise))
    fake_imgs = generator.predict(z)    # 페이크 이미지를 배치사이즈 만큼 만듦

    d_hist_real = discriminator.train_on_batch(real_imgs, real)     # train_on_batch: 에폭 전체토탈 1회만 하고 그만둠/ 민맥에서 학습된 것을 리얼이미지로 받아들임
    d_hist_fake = discriminator.train_on_batch(fake_imgs, fake)     # 페이크 이미지를 주고 학습시킴, 라벨은 페이크. 디스에서 만든 것을 페이크 이미지로 받아들임

    d_loss, d_acc = np.add(d_hist_fake, d_hist_real) * 0.5  # 평균을 구함


    if epoch % 2 == 0:  # 짝수일때/ 제너는 학습이 잘됨=> 그래서 디스보다 절반만 학습시키려고 짝수번에 학습되도록
        z = np.random.normal(0, 1, (batch_size, noise))     # batch_size 10만
        gan_hist = gan_model.train_on_batch(z, real)    # 1이라고 답하게 학습/ 이미지를 간모델에 있는
    # 위에 코드는 학습시킬때 필요한 코드

    # 밑에 코드는 이미지 저장할때 필요하지 학습시키는 것과는 관계가 없음
    if epoch % sample_interval ==0:     # 샘플 이미지 만들어서 저장하는 코드/ 100에폭당 한번씩 이미지 저장
        print('%d, [D loss: %f, acc.: %.2f%%], [G loss: %f]'%(
                epoch, d_loss, d_loss, gan_hist))
        row = col = 4
        z = np.random.normal(0, 1 ,(row*col, noise))
        fake_imgs = generator.predict(z)
        fake_imgs = 0.5 * fake_imgs + 0.5

        _, axs = plt.subplots(row, col, figsize=(5, 5),sharey =True, sharex=True)   # x와 y축을 공유한다/ 스케일 사이즈가 같아진다.
        cont = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(fake_imgs[cont, :, :, 0], cmap='gray') # 회색으로 그리기
                axs[i, j].axis('off')   # 가로축 세로축 없앤다 off
                cont += 1
        path = os.path.join(OUT_DIR, 'img-{}'.format(epoch + 1))
        plt.savefig(path)    # 저장
        plt.close()


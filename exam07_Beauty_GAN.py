import dlib         # 얼굴 인식하는 코드 / 설치 conda install -c conda-forge dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf       # compat 이 빨간줄 뜨는게 정상/ 오래된 버전이라 v1에 맞춘다는 것
tf.disable_v2_behavior()
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


detector = dlib.get_frontal_face_detector() # 앞 얼굴을 찾아주는 detector
shape = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat')   # 다섯개의 랜드마크로 찾음 -> 눈 양쪽 끝, 인중

# 얼굴을 따서 박스 그리는 것
# img = dlib.load_rgb_image('./imgs/02.jpg')
# plt.figure(figsize=(16, 10))
# plt.imshow(img)
# plt.show()
#
# img_result = img.copy()
# dets = detector(img, 1)
#
# if len(dets) == 0:
#     print('Not find faces')
#
# else:
#     fig, ax = plt.subplots(1, figsize=(10, 16)) # subplot (1) 1개 그려라
#     for det in dets:
#         x, y, w, h = det.left(), det.top(), det.width(), det.height()
#         rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='b', facecolor='None')
#         # 이미지 좌표는 무조건 왼쪽 위부터 생성
#         ax.add_patch(rect)
# ax.imshow(img_result)
# plt.show()


# 양쪽 눈 끝과 끝, 인중에 점찍는 것
# fig, ax = plt.subplots(1, figsize=(10,6))
# obj = dlib.full_object_detections()
#
#
# for detection in dets:
#     s = shape(img, detection)
#     obj.append(s)
#
#     for point in s.parts():
#         circle = patches.Circle((point.x, point.y), radius=3, edgecolor='b', facecolor='b')
#         ax.add_patch(circle)
#     ax.imshow(img_result)
# plt.show()


# 얼굴찾아서 정렬하는 함수
def align_face(img):
    dets = detector(img, 1)
    objs = dlib.full_object_detections()
    for detection in dets:
        s = shape(img, detection)
        objs.append(s)
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)  # 패딩 : 0.35만큼 공간을 더 줌
    return faces


# test_faces = align_face(img)
# fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(10, 8))
# axes[0].imshow(img)
# for i, face in enumerate(test_faces):
#     axes[i+1].imshow(face)
# plt.show()


# 모델 불러오기
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
sess.run(init_op)
saver = tf.train.import_meta_graph('./models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

def preprocessing(img):     # 전처리
    return img / 127.5 - 1
def deprocess(img):         # 제너가 생성한 것을 다시 되돌리때 사용
    return 0.5 * img +0.5

img1 = dlib.load_rgb_image('./imgs/no_makeup/vSYYZ306.png')
img1_faces = align_face(img1)

img2 = dlib.load_rgb_image('./imgs/makeup/2020.jpg')     # 이미지를 학습하는게 아니라 그 스타일을 학습함
img2_faces = align_face(img2)

# fig, axes = plt.subplots(1, 2, figsize=(8, 5))
# axes[0].imshow(img1_faces[0])
# axes[1].imshow(img2_faces[0])
# plt.show()

scr_img = img1_faces[0]
ref_img = img2_faces[0]

X_img = preprocessing(scr_img)
X_img = np.expand_dims(X_img, axis=0)

Y_img = preprocessing(ref_img)
Y_img = np.expand_dims(Y_img, axis=0)

output = sess.run(Xs, feed_dict={X:X_img, Y:Y_img})       # xs==제너
output_img = deprocess(output[0])

fig, axes = plt.subplots(1, 3, figsize=(8, 5))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
axes[2].imshow(output_img)
plt.show()


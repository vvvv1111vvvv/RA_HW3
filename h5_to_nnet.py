import tensorflow as tf
from NNet.converters import keras2nnet

# 저장된 .h5 불러오기
model = tf.keras.models.load_model('fashion_mnist.h5')

# 입력 범위(정규화 기준)를 설정해야 합니다.
# MNIST의 경우 [0,1] 범위로 가정할 건데, 필요 시 사전 정규화 파라미터를 지정하세요.
input_bounds = [0.0], [1.0]

# .nnet 파일로 저장
keras2nnet(model,  # 변환할 모델
          input_min=input_bounds[0],
          input_max=input_bounds[1],
          output_min=[0.0],  # 출력 정규화, 필요에 맞게 조정
          output_max=[1.0],
          filename='fashion_mnist.nnet')
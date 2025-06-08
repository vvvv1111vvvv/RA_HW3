# h5_to_onnx_tf2onnx.py
import tensorflow as tf
import tf2onnx



# 입력 경로 및 모델 로딩
keras_model_path = "./resources/keras/cifar10.h5"
onnx_model_path = "./resources/onnx/cifar10/cifar10.onnx"

# 모델 로드
model = tf.keras.models.load_model(keras_model_path)

# 변환
spec = (tf.TensorSpec((None, 32, 32, 3), tf.float32, name="input"),)

# 변환 실행
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_model_path)

print(f"Saved ONNX model to {onnx_model_path}")

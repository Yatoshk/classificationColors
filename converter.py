import tensorflow as tf
import tf2onnx
filePath = 'C:/Users/chesa/OneDrive/Desktop/filesForModel/model/pickcolor.keras'
outputPath = 'C:/Users/chesa/OneDrive/Desktop/filesForModel/model/pickcolor.onnx'
model = tf.keras.models.load_model(filePath)
tf2onnx.convert.from_keras(model, output_path=outputPath)
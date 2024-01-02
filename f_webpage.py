from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load your trained model here
model = tf.keras.models.load_model('model_optimal.h5')

def visualize_features(model, layer_name, img_path):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(img_path)
    num_feature_maps = min(intermediate_output.shape[3], 64)  # Limit the number of feature maps to visualize
    return intermediate_output, num_feature_maps

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_path = "Sample.jpg"
        img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the pixel values

        layer_names = [layer.name for layer in model.layers if 'conv2d' in layer.name]
        features = {}
        for layer_name in layer_names:
            intermediate_output, num_feature_maps = visualize_features(model, layer_name, img_array)
            features[layer_name] = [intermediate_output[0, :, :, i].tolist() for i in range(num_feature_maps)]

        return jsonify(features)
    return render_template('index_plot.html')

if __name__ == '__main__':
    app.run(debug=True)

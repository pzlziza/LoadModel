from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
print(tf.__version__)
print(tf.keras.__version__)
import numpy as np

app = Flask(__name__)

# Load the trained model (ensure 'modelasl.h5' is in the same directory)
model = load_model('c:\cloud-computing\tellme\modelasl.h5')



# Define label names for the classes (A-Z)
label_names = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 
               8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 
               15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 
               22: 'w', 23: 'x', 24: 'y', 25: 'z'}

@app.route('/gesture', methods=['GET'])
def gesture():
    # Check if an image file is provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    
    # Check if the file has a valid filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Load the image and resize it to the expected input size of the model
        img = image.load_img(file, target_size=(71, 71))
        x = image.img_to_array(img) / 255.0  # Normalize the image
        x = np.expand_dims(x, axis=0)  # Add batch dimension

        # Make predictions
        classes = model.predict(x)
        predicted_index = np.argmax(classes[0])  # Get the index of the highest probability
        prediction_name = label_names.get(predicted_index, "Unknown")  # Get the corresponding label
        confidence_percentage = np.max(classes[0]) * 100  # Get the confidence percentage

        return jsonify({
            'prediction': prediction_name,
            'confidence': f"{confidence_percentage:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
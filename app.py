import requests
from flask import Flask, render_template, jsonify, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
MODEL = load_model('mnist_fashion.h5')
ITEMS = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

@app.route('/')
def hello(): 
    return "hello, this is test"
 
@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['gambar']
    img = Image.open(image).convert('L')
    img = img.resize((28,28))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255

    pred = MODEL.predict(img_array)
    # get top 3 predictions indices ordered by probabilty in decending order
    top_n = 3
    indices = np.argpartition(pred, -top_n)[-top_n:]
    indices = np.squeeze(indices)
    indices = np.flip(indices)

    # prepare response
    response_list = [{'item': ITEMS[id], 'probability': str(pred[0][id]), 'label_id': str(id)} for id in indices[:top_n]]
    response_obj = { 'reponse': response_list }

    img.close()
    
    return jsonify(response_obj)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
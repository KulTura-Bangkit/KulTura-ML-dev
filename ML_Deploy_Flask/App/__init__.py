from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import operator

app = Flask(__name__)

#load model
model1 = tf.keras.models.load_model('App/Models/Model_Batik.h5', compile=False)
model2 = tf.keras.models.load_model('App/Models/Model_Makanan.h5')
model3 = tf.keras.models.load_model('App/Models/Model_Rumah.h5', compile= False)

#define label
class_labels1 = ['Batik Cirebon','Batik Ikat Celup','Batik Kalimantan Tengah','Batik Kawung','Batik Papua','Batik Parang','Batik Poleng','Batik Sekar Jagad','Batik Tambal','Batik Truntum']
class_labels2 = ['Apem','Ayam Bakar', 'Bakso','Bakwan', 'Gado Gado','Gudeg','Lumpia','Pempek', 'Rendang', 'Sate']
class_labels3 = ['Bolon', 'Buton', 'Dalam Loka','Gadang', 'Gapura Candi Bentar', 'Honai', 'Joglo', 'Mod Aki Nasa', 'Panjang', 'Tongkonan']

#preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0 
    image = np.expand_dims(image, axis=0)

    return image

#topthree label predictions
def topthree(predictions, labels):
    my_list = labels
    my_list2 = []
    for j in predictions[0]:
        my_list2.append(j)
    my_dict = dict(zip(my_list, my_list2))
    sorted_dict_desc = dict(sorted(my_dict.items(), key=operator.itemgetter(1), reverse=True))
    top_item = list(sorted_dict_desc.keys())[:3]
    top_values = list(sorted_dict_desc.values())[:3]

    top_values = [int(x) for x in top_values]

    output = {'label 1': top_item[0],
              'percent 1': top_values[0],
              'label 2': top_item[1],
              'percent 2': top_values[1] ,
              'label 3': top_item[2],
              'percent 3': top_values[2],
              }

    return output


#Batik Model
@app.route('/predict1', methods=['POST'])
def predict1():

    file = request.files['image']
    
    model1.compile()
    image = Image.open(file)
    processed_image = preprocess_image(image)
    labels = class_labels1
    predictions = model1.predict(processed_image)
    result = topthree(predictions, labels)

    return jsonify(result)

#Traditional Food Model
@app.route('/predict2', methods=['POST'])
def predict2():

    file = request.files['image']
    
    image = Image.open(file)
    processed_image = preprocess_image(image)
    labels = class_labels2
    predictions = model2.predict(processed_image)
    result = topthree(predictions, labels)

    return jsonify(result)

#Traditional House Model
@app.route('/predict3', methods=['POST'])
def predict3():

    file = request.files['image']
    
    model3.compile()
    image = Image.open(file)
    processed_image = preprocess_image(image)
    labels = class_labels3
    predictions = model3.predict(processed_image)
    result = topthree(predictions, labels)

    return jsonify(result)

@app.errorhandler(400)
def not_found(error):
    return jsonify({'title' : 'Bad Request',
                    'status_code': 400,
                    'message': 'Ensure your file extension is jpg, jpeg, or png'})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'title': 'Not Found', 
                    'status_code': 404,
                    'message' :  'Endpoint Not Found'})

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({'title': 'Internal Server Error',
                    'status_code': 500,
                    'message' : 'Ensure your file extension is jpg, jpeg, or png'})



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=500) #disesuaikan di cloud

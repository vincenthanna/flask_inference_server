#!/usr/bin/python3

import io
import os
import ssl
import json
from PIL import Image

import tensorflow as tf
import numpy as np
import math

# models :
from models.model_simple import build_model_simple

from flask import Flask, jsonify, request, session, g, redirect, url_for,\
    abort, render_template, flash


app = Flask(__name__)
app.config.from_object(__name__)
app.debug=True

MODEL_SAVE_DIR_PATH = '../cnn_mnist/trained/'
MODEL_NAME = 'model'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR_PATH, MODEL_NAME)

import io
from PIL import Image
from numpy import asarray

def transform_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = asarray(img.resize((28, 28)))
    img = img.reshape(1, 28, 28 ,1)
    tempimg = img.reshape(28 * 28)
    # 검은색을 255 값으로 반전을 해주어야 한다.
    if tempimg[0] > 200 and tempimg[1]>200:
        img = 255 - img
    return img


# prepare model

classCnt = 10
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, classCnt])
keep_prob = tf.placeholder(tf.float32)

# select model : FIXME:
model = build_model_simple(X, keep_prob=keep_prob, labelCnt=classCnt)


inference_session = tf.Session()
initializer = tf.global_variables_initializer()
inference_session.run(initializer)

saver = tf.compat.v1.train.Saver()
print(">>>>>>>>>>>>>>>>>>> restore previous model")
saver.restore(inference_session, tf.train.latest_checkpoint(MODEL_SAVE_DIR_PATH))


def get_prediction(images):
    ret = inference_session.run(model, feed_dict={X:images.reshape(-1, 28, 28, 1), keep_prob:1.0})
    ret = np.array(ret)
    if len(ret.shape) == 2:
        ret = ret[0]
    retval = np.argmax(ret)
    return retval


# secret key for SSL connection
import secrets
secret_key = secrets.token_urlsafe(32)
app.secret_key = secret_key



# ssl context for HTTPS
ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
ssl_context.load_cert_chain(certfile="localhost.crt", keyfile='localhost.key')







@app.route('/')
def hello():
    return 'Hello World'

@app.route('/predict', methods=['POST'])
def predict():
    print("predict() called")
    if request.method == 'POST':
        file = request.files['file']
        # convert that to bytes
        img_bytes = bytearray(file.read())
        img = transform_image(img_bytes)
        # print(">>>>> img.shape=", img.shape)
        ret = int(get_prediction(images=img))
        return jsonify({'predicted_number' : ret})



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=443, ssl_context=ssl_context)
import io
import os
import ssl
import json
from PIL import Image


from flask import Flask, jsonify, request, session, g, redirect, url_for,\
    abort, render_template, flash

from torchvision import models
import torchvision.transforms as transforms


app = Flask(__name__)
app.config.from_object(__name__)
app.debug=True

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


# prepare model
model = models.densenet121(pretrained=True)
# 모델을 inference에만 사용하도록 설정(eval 모드로 변경)
model.eval()

imagenet_class_index = json.load(open('../_static/imagenet_class_index.json'))

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]




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
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id' : class_id, 'class_name' : class_name})



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=443, ssl_context=ssl_context)
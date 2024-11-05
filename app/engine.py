import os
import json
import numpy as np
import pickle as pk
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from tensorflow.keras.layers import GlobalAveragePooling2D

def fix_flatten_layer(model_path):
    model = load_model(model_path, compile=False)
    config = model.get_config()

    # Remove the batch input shape from Flatten layer
    for layer in config['layers']:
        if layer['class_name'] == 'Flatten' and 'batch_input_shape' in layer['config']:
            del layer['config']['batch_input_shape']

    model = model.__class__.from_config(config)
    model.layers.insert(-1, GlobalAveragePooling2D())
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load models
first_gate = VGG16(weights='imagenet')
print("First gate loaded")

second_gate = fix_flatten_layer(r'C:\Users\saura\Downloads\sahil_project\data1a.h5')
print("Second gate loaded")

location_model = fix_flatten_layer(r'C:\Users\saura\Downloads\sahil_project\data2a\vgg16_damage_location.h5')
print("Location model loaded")

severity_model = fix_flatten_layer(r'C:\Users\saura\Downloads\sahil_project\data3a\vgg16_damage_severity.h5')
print("Severity model loaded")

# Load category list
with open(r'static/models/vgg16_cat_list.pk', 'rb') as f:
    cat_list = pk.load(f)
print("Cat list loaded")

def prepare_img_224(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(f"Prepared image shape for 224x224: {x.shape}")  # Debugging
    return x

def prepare_img_256(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(f"Prepared image shape for 256x256: {x.shape}")  # Debugging
    return x

def car_categories_gate(img_224, model):
    print("Validating if the image is a car...")
    preds = model.predict(img_224)
    top = decode_predictions(preds, top=5)
    print(f"Top predictions: {top}")  # Debugging
    for j in top[0]:
        if j[0:2] in cat_list:
            return True
    return False

def car_damage_gate(img_224, model):
    print("Checking for damage existence...")
    pred = model.predict(img_224)
    print(f"Damage model prediction: {pred}")  # Debugging
    threshold = 0.5  # Assuming binary classification
    if pred[0][0] <= threshold:
        return True
    else:
        return False

def location_assessment(img_224, model):
    print("Assessing damage location...")
    pred = model.predict(img_224)
    pred_label = np.argmax(pred, axis=1)
    d = {0: 'Front', 1: 'Rear', 2: 'Side'}
    print(f"Location prediction: {pred_label}, Label: {d.get(pred_label[0], 'Unknown')}")  # Debugging
    return d.get(pred_label[0], "Unknown")

def severity_assessment(img_224, model):
    print("Assessing damage severity...")
    pred = model.predict(img_224)
    pred_label = np.argmax(pred, axis=1)
    d = {0: 'Minor', 1: 'Moderate', 2: 'Severe'}
    print(f"Severity prediction: {pred_label}, Label: {d.get(pred_label[0], 'Unknown')}")  # Debugging
    return d.get(pred_label[0], "Unknown")

# Main function
def engine(img_path):
    img_224 = prepare_img_224(img_path)
    
    # Gate 1: Validate if it's a car
    g1 = car_categories_gate(img_224, first_gate)
    if not g1:
        print("Failed Gate 1: Not a car image")  # Debugging
        result = {
            'gate1': 'Car validation check: ',
            'gate1_result': 0,
            'gate1_message': 'Are you sure this is a picture of your car? Please retry your submission.',
            'gate2': None,
            'location': None,
            'severity': None,
            'final': 'Damage assessment unsuccessful!'
        }
        return result

    # Gate 2: Check if damage exists
    g2 = car_damage_gate(img_224, second_gate)
    if not g2:
        print("Failed Gate 2: No damage detected")  # Debugging
        result = {
            'gate1': 'Car validation check: ',
            'gate1_result': 1,
            'gate2': 'Damage presence check: ',
            'gate2_result': 0,
            'gate2_message': 'Are you sure that your car is damaged? Please retry your submission.',
            'location': None,
            'severity': None,
            'final': 'Damage assessment unsuccessful!'
        }
        return result

    # Gate 3: Assess damage location
    location = location_assessment(img_224, location_model)

    # Gate 4: Assess damage severity
    severity = severity_assessment(img_224, severity_model)

    # Compile the results
    result = {
        'gate1': 'Car validation check: ',
        'gate1_result': 1,
        'gate2': 'Damage presence check: ',
        'gate2_result': 1,
        'location': location,
        'severity': severity,
        'final': 'Damage assessment complete!'
    }
    return result

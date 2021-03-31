import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Some utilites
import numpy as np
import tokenization
import tensorflow_hub as hub
import lightgbm as lgb
import pickle

# Declare a flask app
app = Flask(__name__)



def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


# Load our own trained model

module_url = "models/bert_en_uncased_L-24_H-1024_A-16_1"

if not os.path.exists(module_url) :
    module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"


print('loading model from internet ...')

bert_layer = hub.KerasLayer(module_url, trainable = True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file = vocab_file, do_lower_case = do_lower_case)

max_len=100

def load_model(bert_layer,max_len=100):
    input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output,sequence_output=bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    model = tf.keras.models.Model(inputs = [input_word_ids, input_mask, segment_ids], outputs = [clf_output])
    LR=0.00001
    model.compile(optimizer = tf.keras.optimizers.Adam(lr = LR),
                  loss = [tf.keras.losses.CategoricalCrossentropy()],
                  metrics = [tf.keras.metrics.Accuracy()])

    return model

model=load_model(bert_layer,max_len)

print('Model loaded. Start serving...')


#Make the prediction from the model
def model_predict(text, model):
    # Preprocessing the Text

    text_array = np.array([text],dtype=np.str)

    text_encoded= bert_encode(text_array, tokenizer, max_len = max_len)

    # Predicting
    preds = model.predict(text_encoded)

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the text from post request
        text = request.json["text"]

        # Classes label
        classes_=np.array(['negative', 'neutral', 'positive'], dtype=np.str)


        # Create Embeddings with Bert
        embeddings = model_predict(text, model)

        # load and predict with lgbm Model
        lgmodel = lgb.Booster(model_file='models/lgbm_model.lgb')
        pred_lgbm = lgmodel.predict(embeddings)


        # load SVM Model
        with open('models/SVM.pickle', 'rb') as f:
            SVM = pickle.load(f)

        # load Scaler
        with open('models/scaler.pickle', 'rb') as f:
            scaler = pickle.load(f)

        #Data standarization
        embeddings_scaled=scaler.transform(embeddings)

        # Predict with SVM
        svm_predict_proba=SVM.predict_proba(embeddings_scaled)

        # Ensemble the results
        Global_predict=(0.6*pred_lgbm+0.4*svm_predict_proba)  #0.4 lgbm 0.6 svm


        #create response
        probabilities={ 'negative': Global_predict[0][0],
                        'neutral': Global_predict[0][1],
                        'positive': Global_predict[0][2]}

        Global_label_predict=classes_[np.argmax(Global_predict,axis=1)]

        return jsonify(result=Global_label_predict[0],probabilities=probabilities)

    return None


if __name__ == '__main__':

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

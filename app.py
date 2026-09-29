import tensorflow as tf
import nltk
import re
import contractions
from nrclex import NRCLex
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.backend import tanh, softmax, dot
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn

from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
from werkzeug.utils import secure_filename
import pickle
import os

# ==================================================================
# IMPORT TEXT EMOTION RECOGNITION MODEL CUSTOM FUNCTIONS & CLASSES
# ==================================================================
def get_emotion_vector(text):
    emo = NRCLex(text)
    scores = emo.raw_emotion_scores

    emo_labels = ["anger", "disgust", "fear", "joy", "anticipation", "sadness", "surprise"]

    vec = np.array([scores.get(e, 0) for e in emo_labels], dtype=float)

    neutral = 1.0 if vec.sum() == 0 else 0.0
    vec = np.append(vec[:-1], neutral)

    # normalize
    if vec.sum() > 0:
        vec = vec / vec.sum()
    return vec

# Multihead Attention layer
@register_keras_serializable()
class RelativeMultiheadAttention(Layer):
  def __init__(self, input_dim, d_model, num_heads, max_len, **kwargs):
    super().__init__(**kwargs)
    self.input_dim = input_dim
    self.d_model = d_model
    self.num_heads = num_heads # number of parallel heads
    self.head_dim = d_model // num_heads # dimensionality per head
    self.max_len = max_len

  def build(self, input_shape):
    # input_dim is automatically grabbed from the previous layer
    input_dim = input_shape[-1]

    self.qkv_layer = Dense(3 * self.d_model)
    self.linear_layer = Dense(self.d_model)

    # Relative Position Bias Table
    self.relative_bias_table = self.add_weight(
        name = "relative_bias_table",
        shape = (2 * self.max_len - 1, self.num_heads),
        initializer = "zeros",
        trainable = True
    )

    # Create distance index matrix
    coords = tf.range(self.max_len)
    relative_coords = coords[None, :] - coords[:, None]
    relative_coords += self.max_len - 1
    self.relative_index = tf.constant(relative_coords)

  def call (self, x, mask = None):
    shape = tf.shape(x)
    batch_size, seq_len, input_dim = shape[0], shape[1], shape[2]

    qkv = self.qkv_layer(x)
    qkv = tf.reshape(qkv, [batch_size, seq_len, self.num_heads, 3 * self.head_dim])
    qkv = tf.transpose(qkv, [0, 2, 1, 3])
    q, k, v = tf.split(qkv, 3, axis = -1)

    attn_scores = tf.matmul(q, k, transpose_b = True)
    attn_scores /= tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

    # Relative Bias Injection
    rel_index = self.relative_index[:seq_len, :seq_len]
    rel_bias = tf.gather(self.relative_bias_table, rel_index)
    rel_bias = tf.transpose(rel_bias, [2, 0, 1])
    attn_scores = attn_scores + tf.expand_dims(rel_bias, 0)

    if mask is not None:
      attn_scores += mask

    attn_weights = tf.nn.softmax(attn_scores, axis = -1)

    out = tf.matmul(attn_weights, v)
    out = tf.transpose(out, [0, 2, 1, 3])
    out = tf.reshape(out, [batch_size, seq_len, self.d_model])

    return self.linear_layer(out)

  def get_config(self):
    config = super().get_config()
    config.update({
        "input_dim": self.input_dim,
        "d_model": self.d_model,
        "num_heads": self.num_heads,
        "max_len": self.max_len,
    })

    return config
  
@register_keras_serializable()
class EmotionGatedLSTMCell(Layer):
    def __init__(self, units, embedding_dim=128, emotion_dim=7, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.embedding_dim = embedding_dim
        self.emotion_dim = emotion_dim
        self.state_size = [units, units]
        self.output_size = units

    def build(self, input_shape):
        total_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(self.embedding_dim + self.units, 4 * self.units),
            initializer='glorot_uniform', name='lstm_kernel')

        self.bias = self.add_weight(
            shape=(4 * self.units,),
            initializer='zeros', name='lstm_bias')

        self.emotion_gate_kernel = self.add_weight(
            shape=(self.emotion_dim, self.units),
            initializer='glorot_uniform', name='emotion_gate_kernel')

        self.alpha = self.add_weight(
            shape=(1,), initializer='ones', name='alpha')

        super().build(input_shape)

    def call(self, inputs, states):
        h_prev, c_prev = states

        x_text = inputs[:, :self.embedding_dim]    
        x_emotion = inputs[:, self.embedding_dim:]  

        # --- 1. Standard LSTM on Text ---
        concat_input = tf.concat([x_text, h_prev], axis=-1)
        gates = tf.matmul(concat_input, self.kernel) + self.bias
        i, f, c_cand, o = tf.split(gates, num_or_size_splits=4, axis=-1)

        f = tf.sigmoid(f + 1.0)
        i = tf.sigmoid(i)
        c_cand = tf.nn.tanh(c_cand)
        o = tf.sigmoid(o)

        # --- 2. NOVELTY: Gate calculation on Emotion Vector ---
        # The gate is calculated purely from the external emotion vector
        gate_activation = tf.nn.tanh(tf.matmul(x_emotion, self.emotion_gate_kernel))

        # --- 3. Update Memory ---
        c_standard = f * c_prev + i * c_cand

        # Inject the emotion gate into the cell memory
        c_new = c_standard + (self.alpha * gate_activation)

        h_new = o * tf.nn.tanh(c_new)

        return h_new, [h_new, c_new]

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))

def preprocess(text):
    # Fix contractions (e.g., "don't" -> "do not")
    text = contractions.fix(text.lower())

    # Remove non-alphabetical characters
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in text.split()]

    return ' '.join(tokens).strip()
    
# ==================================================================
# IMPORT SPEECH EMOTION RECOGNITION MODEL CUSTOM FUNCTIONS & CLASSES
# ==================================================================

def emotion_change_reg(time_step_logits):
    
    # calculate differencee between t-1 & t time steps
    diff = time_step_logits[:, 1:, :] - time_step_logits[:, :-1, :]

    loss = torch.mean(diff **2) # squared 'diff' and get the mean over Batch,time,class

    return loss

class CRNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # --------------------
        # CNN
        # --------------------
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),     
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),        
            nn.Dropout(0.3),
        )

        self.time_steps = 25
        self.base_feat_dim = 64 * 30      

        # --------------------
        # LSTM
        # --------------------
        self.lstm = nn.LSTM(
            input_size=self.base_feat_dim,
            hidden_size=128,
            batch_first=True
        )

        # --------------------
        # Classifier
        # --------------------
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, 128, 108)

        # CNN
        x = self.cnn(x)                        

        # CNN → LSTM (reshape)
        x = x.permute(0, 3, 1, 2)               
        x = x.reshape(x.size(0), x.size(1), -1)   #

        # LSTM
        lstm_out, _ = self.lstm(x)                      

        # Frame-wise classification
        time_logit = self.classifier(lstm_out)      

        # Utterance-level prediction (language express)
        language_logit = time_logit.mean(dim=1)     

        return time_logit, language_logit


app = Flask(__name__)

# ====================================
# TEXT EMOTION MODEL
# ====================================

MAX_LEN = 100

text_emotion_model = load_model(
   "combined_final.keras", 
   custom_objects = {"Attention": RelativeMultiheadAttention, 
                     "EmotionGatedLSTMCell": EmotionGatedLSTMCell}
)

with open("tokenizer.pkl", "rb") as f: 
    tokenizer = pickle.load(f)

with open("emoji_rules.pkl", "rb") as e: 
   emoji_rules = pickle.load(e)

text_label_map = {
    0: "Angry", 
    1: "Disgust", 
    2: "Fear", 
    3: "Happy", 
    4: "Neutral", 
    5: "Sad", 
    6: "Surprise"
}

# ====================================
# SPEECH EMOTION MODEL
# ====================================
speech_emotion_model = CRNN()
speech_emotion_model.load_state_dict(torch.load("TR_CRNN.pth", map_location = "cpu"))

speech_emotion_model.eval()

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok = True)

# ====================================
# ROUTES
# ====================================

@app.route("/")
def home(): 
    return render_template("index.html")

@app.route("/analyze_text", methods = ["POST"])
def analyze_text(): 
    raw_text = request.json.get("text")

    for char in raw_text: 
       if char in emoji_rules: 
          emoji_count = sum(1 for c in raw_text if c in emoji_rules)
          confidence = min(60 + (emoji_count * 10), 90)

          return jsonify({
             "emotion": emoji_rules[char], 
             "confidence": confidence
          })
       
    text = preprocess(raw_text)
    
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        seq, 
        maxlen = MAX_LEN, 
        padding = "pre", 
        truncating = "post"
    )
    
    context_vec = get_emotion_vector(text).reshape(1, 7)

    preds = text_emotion_model.predict([padded, context_vec])

    emotion_id = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    response = {
       "emotion": text_label_map[emotion_id], 
       "confidence": round(confidence * 100, 2)
    }
    return jsonify(response)

@app.route("/analyze_speech", methods = ["POST"])
def analyze_speech(): 
    audio_file = request.files["audio"]

    filename = secure_filename(audio_file.filename)
    audio_path = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(audio_path)
    
    # Load Audio
    y, sr = librosa.load(
       audio_path, 
       sr = 22050, 
       duration = 2.5, 
       offset = 0.6, 
       mono = True
    )

    # Extract Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y = y, 
        sr = sr, 
        n_fft = 2048, 
        hop_length = 512, 
        n_mels = 128
    )

    # Convert to Log Scale
    log_mel = librosa.power_to_db(mel, ref = np.max)

    # Normalize
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)

    # Ensure fixed time dimension
    if log_mel.shape[1] < 108: 
       log_mel = np.pad(log_mel, ((0, 0), (0, 108 - log_mel.shape[1])))

    else: 
       log_mel = log_mel[:, :108]

    # Reshape for CNN
    log_mel = np.expand_dims(log_mel, axis = -1)
    log_mel = np.expand_dims(log_mel, axis = 0) 

    with torch.no_grad(): 
       x = torch.tensor(log_mel, dtype = torch.float32).permute(0, 3, 1, 2)
       _, logits = speech_emotion_model(x)
       probs = torch.softmax(logits, dim = 1).cpu().numpy()

    # logits can be negative, greater than 1 (unbounded)

    emotion_id = int(np.argmax(probs))
    confidence = float(probs[0][emotion_id])

    speech_label_map = {
        0: "Angry", 
        1: "Disgust", 
        2: "Fear", 
        3: "Happy", 
        4: "Neutral", 
        5: "Sad", 
        6: "Surprise"
    }

    return jsonify({
        "emotion": speech_label_map[emotion_id], 
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__": 
    app.run(debug = True)
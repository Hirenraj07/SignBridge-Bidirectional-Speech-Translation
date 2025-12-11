import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import json
import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# --- CONFIGURATION ---
MODEL_SAVE_PATH = 'sign_language_model.h5'
TOKENIZER_SAVE_PATH = 'tokenizer.json'
MAX_POSE_LENGTH = 150
LSTM_UNITS = 256

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Sign Language Interpreter",
    page_icon="🤟",
    layout="wide"
)

# --- FIXED DATASET PATH ---
@st.cache_resource
def download_and_get_path():
    """Return the manually set dataset path (KaggleHub cached)."""
    path = r"C:\Users\hiren\.cache\kagglehub\datasets\risangbaskoro\wlasl-processed\versions\5"
    if os.path.exists(path):
        print(f"✅ Using cached dataset path: {path}")
        return path
    else:
        st.error(f"❌ Dataset not found at {path}")
        return None

dataset_path = download_and_get_path()

# --- DEBUG BLOCK to show folder structure ---
if dataset_path:
    st.subheader("DEBUG: Dataset Folder Contents")
    st.write(f"Dataset path: `{dataset_path}`")
    try:
        folder_contents = os.listdir(dataset_path)
        st.write(folder_contents)
    except Exception as e:
        st.error(f"Could not list directory contents: {e}")

# --- LOAD ML COMPONENTS ---
@st.cache_resource
def load_ml_components():
    print("Loading ML components...")
    model = load_model(MODEL_SAVE_PATH)
    with open(TOKENIZER_SAVE_PATH, 'r', encoding='utf-8') as f:
        tokenizer = tokenizer_from_json(json.load(f))
    
    encoder_inputs = model.input[0]
    _, state_h_enc, state_c_enc = model.get_layer('encoder_lstm').output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_lstm_layer = model.get_layer('decoder_lstm')
    decoder_embedding_layer = model.get_layer('decoder_embedding')
    decoder_dense_layer = model.get_layer('decoder_dense')

    decoder_inputs_inf = Input(shape=(1,))
    decoder_state_input_h = Input(shape=(LSTM_UNITS,))
    decoder_state_input_c = Input(shape=(LSTM_UNITS,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    embedded_decoder_inputs_inf = decoder_embedding_layer(decoder_inputs_inf)
    decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm_layer(
        embedded_decoder_inputs_inf, initial_state=decoder_states_inputs)
    decoder_states_inf = [state_h_inf, state_c_inf]
    decoder_outputs_inf = decoder_dense_layer(decoder_outputs_inf)
    
    decoder_model = Model(
        [decoder_inputs_inf] + decoder_states_inputs,
        [decoder_outputs_inf] + decoder_states_inf)

    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    print("✅ ML components loaded successfully.")
    return encoder_model, decoder_model, tokenizer, reverse_word_index

try:
    encoder_model, decoder_model, sl_tokenizer, reverse_word_index = load_ml_components()
    models_loaded = True
except FileNotFoundError:
    st.error(f"Model files not found! Make sure '{MODEL_SAVE_PATH}' and '{TOKENIZER_SAVE_PATH}' are in the same directory as this script.")
    models_loaded = False

# --- LOAD VIDEO DICTIONARY ---
@st.cache_data
def load_video_dictionary(_dataset_path):
    if not _dataset_path:
        return {}
    json_path = os.path.join(_dataset_path, 'WLASL_v0.3.json')
    print("Loading video dictionary...")
    with open(json_path) as f:
        data = json.load(f)
    video_dict = {item['gloss']: [instance['video_id'] for instance in item['instances']] for item in data}
    print("✅ Video dictionary loaded.")
    return video_dict

video_dict = load_video_dictionary(dataset_path)

# --- SIGN TO TEXT TRANSFORMER ---
class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.sequence = []
        self.current_prediction = ""

    def _translate_sequence(self, input_seq):
        states_value = encoder_model.predict(input_seq, verbose=0)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sl_tokenizer.word_index['<start>']
        stop_condition, decoded_sentence = False, ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = reverse_word_index.get(sampled_token_index, '')
            if (sampled_word == '<end>' or sampled_word == '' or len(decoded_sentence.split()) > 20):
                stop_condition = True
            else:
                decoded_sentence += ' ' + sampled_word
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]
        return decoded_sentence.strip()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image)
        
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        combined_landmarks = np.concatenate([pose, lh, rh])
        
        self.sequence.append(combined_landmarks)
        self.sequence = self.sequence[-MAX_POSE_LENGTH:]

        if len(self.sequence) == MAX_POSE_LENGTH:
            input_data = np.expand_dims(np.array(self.sequence, dtype=np.float32), axis=0)
            self.current_prediction = self._translate_sequence(input_data)
            self.sequence = [] 

        cv2.rectangle(img, (0, img.shape[0] - 50), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        cv2.putText(img, self.current_prediction, (10, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        return img

# --- STREAMLIT UI ---
st.title("🤟 Bidirectional Sign Language Interpreter")

if models_loaded and dataset_path and video_dict:
    tab1, tab2 = st.tabs(["Sign to Text (Live)", "Text to Sign (Video Dictionary)"])

    # --- TAB 1: Real-time Sign to Text ---
    with tab1:
        st.header("Real-Time Sign Language to Text")
        st.write("Enable your webcam below and start signing. The AI will translate it into text.")
        webrtc_streamer(
            key="sign-to-text", 
            video_processor_factory=SignLanguageTransformer,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

    # --- TAB 2: Text to Sign Video Dictionary ---
    with tab2:
        st.header("Text to Sign Video Dictionary")
        st.write("Enter a word to see a video of the corresponding sign from the WLASL dataset.")
        
        search_word = st.text_input("Enter a word:", key="text-to-sign-input").lower()

        if st.button("Find Sign Video", key="text-to-sign-button"):
            if search_word in video_dict:
                st.success(f"Found videos for '{search_word}'!")
                video_id_to_show = video_dict[search_word][0]

                # ✅ Correct KaggleHub path
                video_path = os.path.join(dataset_path, 'videos', f"{video_id_to_show}.mp4")
                st.write(f"DEBUG: Looking for video at `{video_path}`")

                # --- FIX: If video not found, look for nearby matches (like 27172 instead of 27171)
                if not os.path.exists(video_path):
                    import glob
                    st.warning(f"Original video not found for ID {video_id_to_show}. Searching for nearby matches...")
                    all_videos = glob.glob(os.path.join(dataset_path, "videos", "*.mp4"))
                    similar_videos = [
                        v for v in all_videos
                        if str(video_id_to_show)[:4] in os.path.basename(v)
                    ]
                    if similar_videos:
                        video_path = similar_videos[0]
                        st.info(f"Showing nearby video: {os.path.basename(video_path)}")
                    else:
                        st.error("❌ No nearby videos found either. Check your dataset folder.")

                # --- Display video if it exists ---
                if os.path.exists(video_path):
                    st.video(video_path)
                else:
                    st.error(f"❌ Video file not found at path: {video_path}")
                    st.info("Make sure this file exists in your KaggleHub cache.")
            else:
                st.warning(f"The word '{search_word}' was not found in the dataset's vocabulary.")

else:
    st.header("Project Not Ready")
    st.write("Please make sure all required files (`sign_language_model.h5`, `tokenizer.json`, `kaggle.json`) are present and the dataset can be downloaded.")

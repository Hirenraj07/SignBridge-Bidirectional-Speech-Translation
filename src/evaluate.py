import os
import json
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

# Import functions from your main script
# Ensure your main.py file is in the same directory
from main import download_dataset, MAX_POSE_LENGTH, MAX_LABEL_LENGTH, LSTM_UNITS

# --- CONFIGURATION ---
MODEL_SAVE_PATH = 'sign_language_model.h5'
TOKENIZER_SAVE_PATH = 'tokenizer.json'
DATASET_NAME = "risangbaskoro/wlasl-processed"
# We will only process this many videos to create our test set - MUCH FASTER!
MAX_VIDEOS_TO_EVALUATE = 300

# --- Re-define process_videos here to include the limit ---
def process_videos_for_eval(dataset_path):
    json_path = os.path.join(dataset_path, 'WLASL_v0.3.json')
    videos_dir = os.path.join(dataset_path, 'videos')
    with open(json_path) as f: data = json.load(f)
    all_poses, all_labels = [], []
    
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for i, item in enumerate(tqdm(data, desc="Processing videos for evaluation")):
            if i >= MAX_VIDEOS_TO_EVALUATE:
                break
            label = item['gloss']
            for instance in item['instances']:
                video_id = instance['video_id']
                video_path = os.path.join(videos_dir, f"{video_id}.mp4")
                if not os.path.exists(video_path): continue

                cap = cv2.VideoCapture(video_path)
                landmarks_per_video = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    
                    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
                    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
                    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
                    
                    landmarks_per_video.append(np.concatenate([pose, lh, rh]))
                cap.release()
                
                if landmarks_per_video:
                    all_poses.append(np.array(landmarks_per_video))
                    all_labels.append(label)
    return all_poses, all_labels

# (The other functions like load_components, build_inference_models, and translate_sequence are the same)
# We will just copy them here for a self-contained script.

def load_components():
    print("Loading model and tokenizer...")
    model = load_model(MODEL_SAVE_PATH)
    with open(TOKENIZER_SAVE_PATH, 'r', encoding='utf-8') as f:
        tokenizer = tokenizer_from_json(json.load(f))
    return model, tokenizer

def build_inference_models(model):
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
    return encoder_model, decoder_model

def translate_sequence(input_seq, encoder_model, decoder_model, tokenizer):
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    states_value = encoder_model.predict(input_seq, verbose=0)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']
    stop_condition, decoded_sentence = False, ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_word_index.get(sampled_token_index, '')
        if (sampled_word == '<end>' or sampled_word == '' or len(decoded_sentence.split()) > MAX_LABEL_LENGTH):
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return decoded_sentence.strip()


# --- MAIN EVALUATION SCRIPT ---
if __name__ == "__main__":
    model, tokenizer = load_components()
    
    print(f"\nQuick evaluation mode: Processing {MAX_VIDEOS_TO_EVALUATE} videos to create a test set...")
    dataset_path = download_dataset(DATASET_NAME)
    all_poses, all_labels = process_videos_for_eval(dataset_path)

    tokenized_labels = tokenizer.texts_to_sequences(all_labels)
    tokenized_labels = [ [tokenizer.word_index['<start>']] + seq + [tokenizer.word_index['<end>']] for seq in tokenized_labels ]
    
    padded_poses = tf.keras.preprocessing.sequence.pad_sequences(all_poses, maxlen=MAX_POSE_LENGTH, padding='post', truncating='post', dtype='float32')
    padded_labels = tf.keras.preprocessing.sequence.pad_sequences(tokenized_labels, maxlen=MAX_LABEL_LENGTH, padding='post', truncating='post')
    
    # We'll use a simple 80/20 split on this smaller dataset just for evaluation
    _, X_test, _, y_test = train_test_split(padded_poses, padded_labels, test_size=0.2, random_state=42)
    print(f"Test set created with {len(X_test)} samples.")

    # (The rest of the evaluation is the same as before)
    print("\n--- 1. Keras Model Evaluation ---")
    decoder_input_test = y_test[:, :-1]
    decoder_target_test = y_test[:, 1:]
    loss, accuracy = model.evaluate([X_test, decoder_input_test], decoder_target_test, verbose=0)
    print(f"Test Set Loss: {loss:.4f}")
    print(f"Test Set Accuracy: {accuracy*100:.2f}%")

    print("\n--- 2. Qualitative Evaluation (Sample Translations) ---")
    encoder_model, decoder_model = build_inference_models(model)
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    for i in np.random.choice(len(X_test), 5):
        input_seq = X_test[i:i+1]
        predicted_sentence = translate_sequence(input_seq, encoder_model, decoder_model, tokenizer)
        actual_sentence = ' '.join([reverse_word_index.get(token, '?') for token in y_test[i] if token > 0 and token not in [tokenizer.word_index['<start>'], tokenizer.word_index['<end>']]])
        print('-'*50)
        print("Actual:   ", actual_sentence)
        print("Predicted:", predicted_sentence)
    print('-'*50)

    print("\n--- 3. BLEU Score Evaluation (Translation Quality) ---")
    actuals, predictions = [], []
    for i in tqdm(range(len(X_test)), desc="BLEU Score Preds"):
        input_seq = X_test[i:i+1]
        predicted_sentence = translate_sequence(input_seq, encoder_model, decoder_model, tokenizer)
        actual_sentence = ' '.join([reverse_word_index.get(token, '?') for token in y_test[i] if token > 0 and token not in [tokenizer.word_index['<start>'], tokenizer.word_index['<end>']]])
        actuals.append([actual_sentence.split()])
        predictions.append(predicted_sentence.split())

    bleu_score = corpus_bleu(actuals, predictions)
    print(f"\nCorpus BLEU Score on Test Set: {bleu_score:.4f}")
import numpy as np
import cv2
import streamlit as st
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import matplotlib.pyplot as plt

# Define the emotions.
emotion_labels = ['neutral', 'satisfied']

# Load model.
classifier = load_model('model_78.h5')
classifier.load_weights("model_weights_78.h5")

# Load face using OpenCV
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

# Initialize emotion counters.
emotion_counts = {'neutral': 0, 'satisfied': 0}


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.emotion_counts = {'neutral': 0, 'satisfied': 0}

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_labels[maxindex]
                output = str(finalout)
                self.emotion_counts[output] += 1

            label_position = (x, y - 10)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img


def plot_emotion_distribution(emotion_counts, title="Emotion Distribution"):
    labels = list(emotion_counts.keys())
    values = list(emotion_counts.values())

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title)
    st.pyplot(fig)


def summarize_emotions(saved_data):
    total_counts = {'neutral': 0, 'satisfied': 0}

    for data in saved_data:
        total_counts['neutral'] += data['neutral']
        total_counts['satisfied'] += data['satisfied']

    total = sum(total_counts.values())
    neutral_percentage = (total_counts['neutral'] / total) * 100 if total > 0 else 0
    satisfied_percentage = (total_counts['satisfied'] / total) * 100 if total > 0 else 0

    st.write("### Summary of Emotion Distribution")
    st.write(f"Total Neutral: {total_counts['neutral']} ({neutral_percentage:.1f}%)")
    st.write(f"Total Satisfied: {total_counts['satisfied']} ({satisfied_percentage:.1f}%)")

    return total_counts


def main():
    st.title("Kuro Real Time Satisfaction Detection")
    st.write(
        "This application is designed to capture and analyze real-time emotion data from a field area filled with people. Using advanced facial recognition technology, the application identifies and classifies the emotions of individuals in the crowd into two categories: 'neutral' and 'satisfied'. The primary objective is to provide insightful analytics on the emotional state of the crowd during an event.")
    st.write("1. Click Start to open your camera and give permission for recording.")
    st.write("2. Click 'Save Current Data' to save the current emotion data.")
    st.write("3. Click 'Compare Saved Data' to compare saved emotion distributions.")
    st.write("4. Click 'Reset Saved Data' to clear all saved data.")
    st.write("5. Click 'Graph Calculation' to display the current emotion distribution.")
    st.write("6. Click 'Summarize Emotions' to get a summary of all saved emotion data.")
    st.write("7. When you're done, click Stop to end.")

    if 'saved_data' not in st.session_state:
        st.session_state.saved_data = []

    ctx = webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

    if ctx.video_processor:
        if st.button('Graph Calculation'):
            plot_emotion_distribution(ctx.video_processor.emotion_counts, title="Current Emotion Distribution")

        if st.button('Save Current Data'):
            st.session_state.saved_data.append(ctx.video_processor.emotion_counts.copy())
            st.success("Current emotion data saved!")

        if st.button('Compare Saved Data'):
            if len(st.session_state.saved_data) < 2:
                st.warning("You need at least two saved data sets to compare.")
            else:
                num_plots = len(st.session_state.saved_data)
                fig, axs = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))

                # Ensure axs is always a list
                if num_plots == 1:
                    axs = [axs]

                for i, data in enumerate(st.session_state.saved_data):
                    labels = list(data.keys())
                    values = list(data.values())
                    axs[i].pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                    axs[i].axis('equal')
                    axs[i].set_title(f"Saved Emotion Distribution {i + 1}")

                st.pyplot(fig)

        if st.button('Reset Saved Data'):
            st.session_state.saved_data = []
            st.success("All saved data has been reset.")

        if st.button('Summarize Emotions'):
            total_counts = summarize_emotions(st.session_state.saved_data)
            plot_emotion_distribution(total_counts, title="Overall Emotion Distribution")


if __name__ == "__main__":
    main()

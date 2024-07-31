import numpy as np
import cv2
import streamlit as st
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import matplotlib.pyplot as plt
import time
import av
from datetime import datetime
import csv
import io
import pandas as pd

# Define the labels
emotion_labels = ['neutral', 'satisfied', 'unsatisfied']

# Load model
try:
    classifier = load_model('kuro_model.h5', compile=False)
    classifier.load_weights("kuro_model_weights.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Load face using OpenCV
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

# Store counts over time
emotion_counts_over_time = {'neutral': [], 'satisfied': [], 'unsatisfied': []}
timestamps = []

class VideoTransformer(VideoTransformerBase):
    # Process every 3rd frame for reducing load
    def __init__(self):
        self.emotion_counts = {'neutral': 0, 'satisfied': 0, 'unsatisfied': 0}
        self.frame_count = 0
        self.skip_frames = 1
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)
        if self.frame_count % self.skip_frames != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Convert to grayscale for face detection
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=3)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Prediction
                prediction = classifier.predict(roi, verbose=0)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_labels[maxindex]
                output = str(finalout)
                self.emotion_counts[output] += 1

                label_position = (x, y - 10)
                cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def plot_real_time_emotion_distribution(timestamps, emotion_counts_over_time, placeholder):
    fig, ax = plt.subplots()
    for emotion in emotion_counts_over_time:
        ax.plot(timestamps, emotion_counts_over_time[emotion], label=emotion)
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    plt.title("Real-Time Emotion Distribution")
    placeholder.pyplot(fig)
    plt.close(fig)

def plot_emotion_distribution(emotion_counts, title):
    labels = list(emotion_counts.keys())
    values = list(emotion_counts.values())
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.title(title)
    st.pyplot(fig)
    plt.close(fig)

def summarize_emotions(saved_data):
    total_counts = {'neutral': 0, 'satisfied': 0, 'unsatisfied': 0}
    for data in saved_data:
        for emotion, count in data["counts"].items():
            total_counts[emotion] += count
    return total_counts

def calculate_percentages(counts):
    total = sum(counts.values())
    percentages = {emotion: (count / total) * 100 for emotion, count in counts.items()}
    return percentages

# Saved data to CSV format
def convert_to_csv(saved_data):
    csv_data = []
    for data in saved_data:
        counts = data["counts"]
        percentages = calculate_percentages(counts)
        csv_data.append({
            'start_time': data['start_time'],
            'end_time': data['end_time'],
            'neutral': f"{percentages['neutral']:.2f}%",
            'satisfied': f"{percentages['satisfied']:.2f}%",
            'unsatisfied': f"{percentages['unsatisfied']:.2f}%"
        })
    return csv_data
def download_csv(csv_data, file_name="saved_emotion_data.csv"):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=csv_data[0].keys())
    writer.writeheader()
    writer.writerows(csv_data)
    return output.getvalue()

def main():
    st.title("Kuro Real Time Satisfaction Detection")
    st.write(
        "This application is designed to capture and analyze real-time emotion data from a field area filled with people. Using advanced facial recognition technology, the application identifies and classifies the emotions of individuals in the crowd into three categories: 'neutral', 'satisfied', and 'unsatisfied'. The primary objective is to provide insightful analytics on the emotional state of the crowd during an event.")
    st.write("1. Start the camera by clicking the 'Start' button and give permission for recording.")
    st.write("2. Click 'Save Current Data' to save the current emotion data.")
    st.write("3. Click 'Compare Saved Data' to compare saved emotion distributions.")
    st.write("4. Click 'Reset Saved Data' to clear all saved data.")
    st.write("5. Click 'Graph Calculation' to display the current emotion distribution.")
    st.write("6. Click 'Summarize Emotions' to get a summary of all saved emotion data.")
    st.write("7. When you're done, click Stop to end.")

    if 'saved_data' not in st.session_state:
        st.session_state.saved_data = []

    ctx = webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

    plot_placeholder = st.empty()

    if 'start_time' not in st.session_state:
        st.session_state.start_time = None

    if 'end_time' not in st.session_state:
        st.session_state.end_time = None

    # Display control buttons
    st.write("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <style>
            .stButton>button {
                width: 100%;
            }
            </style>
            """, unsafe_allow_html=True)
        if st.button('Start Real-Time Detection'):
            if ctx.video_processor:
                st.session_state.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                while True:
                    if not ctx.video_processor:
                        break
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    timestamps.append(current_time)
                    emotion_counts_over_time['neutral'].append(ctx.video_processor.emotion_counts['neutral'])
                    emotion_counts_over_time['satisfied'].append(ctx.video_processor.emotion_counts['satisfied'])
                    emotion_counts_over_time['unsatisfied'].append(ctx.video_processor.emotion_counts['unsatisfied'])
                    plot_real_time_emotion_distribution(timestamps, emotion_counts_over_time, plot_placeholder)
                    time.sleep(1)  # Pause for 1 second to avoid rapid plotting and high CPU usage

        if st.button('Save Current Data'):
            if ctx.video_processor:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                st.session_state.end_time = timestamp
                st.session_state.saved_data.append({
                    "start_time": st.session_state.start_time,
                    "end_time": st.session_state.end_time,
                    "counts": ctx.video_processor.emotion_counts.copy()
                })
                st.success(f"Current emotion data saved! Start Time: {st.session_state.start_time}, End Time: {st.session_state.end_time}")
                st.session_state.start_time = None  # Reset start time for next session
            else:
                st.warning("Video processor not initialized.")

        if st.button('Compare Saved Data'):
            if len(st.session_state.saved_data) < 2:
                st.warning("You need at least two saved data sets to compare.")
            else:
                st.session_state.compare_trigger = True

    with col2:
        st.markdown(
            """
            <style>
            .stButton>button {
                width: 100%;
            }
            </style>
            """, unsafe_allow_html=True)
        if st.button('Reset Saved Data'):
            st.session_state.saved_data = []
            st.success("All saved data has been reset.")

        if st.button('Summarize Emotions'):
            st.session_state.summarize_trigger = True

        if st.button('Graph Calculation'):
            st.session_state.graph_trigger = True

    st.write("---")

    if 'compare_trigger' in st.session_state and st.session_state.compare_trigger:
        st.session_state.compare_trigger = False
        num_plots = len(st.session_state.saved_data)
        fig, axs = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))

        # Ensure axs is always a list
        if num_plots == 1:
            axs = [axs]

        for i, data in enumerate(st.session_state.saved_data):
            labels = list(data["counts"].keys())
            values = list(data["counts"].values())
            axs[i].pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            axs[i].axis('equal')
            axs[i].set_title(
                f"Saved Emotion Distribution {i + 1} \nStart Time: {data['start_time']} \nEnd Time: {data['end_time']}")

        st.pyplot(fig)
        plt.close(fig)  # Close the figure after rendering

    if 'summarize_trigger' in st.session_state and st.session_state.summarize_trigger:
        st.session_state.summarize_trigger = False
        if st.session_state.saved_data:
            total_counts = summarize_emotions(st.session_state.saved_data)
            percentages = calculate_percentages(total_counts)
            plot_emotion_distribution(percentages, title="Overall Emotion Distribution")

            # Determine the emotion with the highest percentage
            most_common_emotion = max(percentages, key=percentages.get)
            st.write(f"### Overall Emotion Distribution Percentages: {most_common_emotion.capitalize()}")
            st.write(
                f"Counts: neutral: {percentages['neutral']:.2f}%, satisfied: {percentages['satisfied']:.2f}%, unsatisfied: {percentages['unsatisfied']:.2f}%")

            # Display a pop-up notification with the most common emotion
            if most_common_emotion == 'satisfied':
                st.success("Most of Emotion is Satisfied 😍")
            elif most_common_emotion == 'neutral':
                st.info("Most of Emotion is Neutral 😐")
            elif most_common_emotion == 'unsatisfied':
                st.warning("Most of Emotion is Unsatisfied 😟")

            # Display the saved data with timestamps and counts
            st.write("### Detailed Emotion Data with Timestamps")
            for i, data in enumerate(st.session_state.saved_data):
                st.write(f"#### Saved Data {i + 1}")
                st.write(f"Start Time: {data['start_time']}")
                st.write(f"End Time: {data['end_time']}")
                data_percentages = calculate_percentages(data["counts"])
                st.write(
                    f"Counts: neutral: {data_percentages['neutral']:.2f}%, satisfied: {data_percentages['satisfied']:.2f}%, unsatisfied: {data_percentages['unsatisfied']:.2f}%")
                st.write("---")
        else:
            st.warning("No saved emotion data to summarize.")

    # function trigger graph calculation
    if 'graph_trigger' in st.session_state and st.session_state.graph_trigger:
        st.session_state.graph_trigger = False
        if ctx.video_processor:
            emotion_counts = ctx.video_processor.emotion_counts
            percentages = calculate_percentages(emotion_counts)
            plot_emotion_distribution(percentages, title="Real-Time Emotion Distribution")
        else:
            st.warning("Video processor not initialized.")

    # Display saved data table
    if st.session_state.saved_data:
        csv_data = convert_to_csv(st.session_state.saved_data)
        df = pd.DataFrame(csv_data)
        st.write("### Saved Data")
        st.table(df)

    st.write("### Export Data")

    if st.button("Export Data to CSV"):
        if st.session_state.saved_data:
            csv_data = convert_to_csv(st.session_state.saved_data)
            csv = download_csv(csv_data)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="saved_emotion_data.csv",
                mime="text/csv",
            )
            st.success("CSV file ready for download.")
        else:
            st.warning("No data to export.")


if __name__ == "__main__":
    main()

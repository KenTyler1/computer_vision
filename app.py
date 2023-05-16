from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from time import sleep

# Load pre-trained model and categories
model = keras.models.load_model("mobilenet.h5")
categories = ['annona','apples','bananas','lemons', 'mango', 'oranges', 'tomatoes', 'human', 'pen', 'phone' ]

# Initialize video capture
cap = cv2.VideoCapture(0)
sleep(2) # Warm-up camera

# Define colors for bounding box and text
color = (0, 255, 0) # Green bounding box
text_color = (255, 255, 255) # White text

# Initialize bounding box dimensions and class detected
xmin, ymin, xmax, ymax = 0, 0, 0, 0
class_detected = False

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/xinchao')
def XinChaoMoiNguoi():
    return "<h3>Xin chào mọi người</h3>"

def gen():
    cap = cv2.VideoCapture(0)

    # Loop to capture and process video frames
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Resize frame to reduce processing time
        frame_resized = cv2.resize(frame, (128, 128))

        # Preprocess image
        frame_preprocessed = preprocess_input(frame_resized)

        # Predict class probabilities
        predictions = model.predict(np.array([frame_preprocessed]))

        # Get index of predicted class with highest probability
        pred_index = np.argmax(predictions)

        # Get name and probability of predicted class
        pred_name = categories[pred_index]
        pred_prob = round(predictions[0][pred_index] * 100, 2)

        # Find contours in frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find contour with largest area
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        # Draw bounding box around largest contour
        if max_contour is not None:
            # Get bounding box dimensions
            x,y,w,h = cv2.boundingRect(max_contour)
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h
            # Check if predicted class is in categories
            if pred_name in categories:
                # Draw bounding box with red color
                color = (0, 0, 255)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness=2)
            else:
                # Draw bounding box with green color
                color = (0, 255, 0)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness=2)


            # Draw text with predicted class name and probability
            text = f"{pred_name} ({pred_prob}%)"
            cv2.putText(frame, text, (xmin+5, ymin+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

            # Display top 3 predictions with highest probabilities
            top3_indices = np.argsort(predictions[0])[::-1][:3]
            for i, idx in enumerate(top3_indices):
                if categories[idx] == pred_name:
                    continue
                text = f"{categories[idx]}: {round(predictions[0][idx]*100,2)}%"
                cv2.putText(frame, text, (10, 20*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)


        cv2.imwrite('demo.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b"\r\n" )

@app.route('/mo_webcam')
def VideoTheoThuMuc():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame" )


if __name__ == '__main__':
    app.run()

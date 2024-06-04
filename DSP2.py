import cv2
import numpy as np
from keras.models import model_from_json
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Error handling for imports
try:
    import cv2
    import numpy as np
    from keras.models import model_from_json
    import streamlit as st
    from PIL import Image
    from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
except ImportError as e:
    st.error(f"Error importing module: {e}")

# Load your trained model with error handling
try:
    with open('model-vgg19.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('model-vgg19.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define class labels
class_labels = {
    0: "crazing",
    1: "inclusion",
    2: "no_defect",
    3: "patches",
    4: "pitted_surface",
    5: "rolled_in_scale",
    6: "scratches"
}

# Streamlit UI
st.title('Identification & Classification of Stainless Steel Defect Types')

# Display instructions
st.markdown("""
    <div style="width:100%; overflow-x:auto;">
        <p style="text-align: left; font-size: 15px;">1) Prepare an image of stainless steel defect types and display it on phone.</p>
        <p style="text-align: left; font-size: 15px;">2) Make sure device's camera is turned on and allowed access.</p>
        <p style="text-align: left; font-size: 15px;">3) Align the image with the green frame shown in the camera below.</p>
        <p style="text-align: left; font-size: 15px;">4) "Predicted" is the result of the defect type identified and classified by the camera.</p>
        <p style="text-align: left; font-size: 15px;">5) "Confidence" is the confidence given by the camera.</p>
        <p style="text-align: left; font-size: 15px;">6) The closer the value to 1.00, the more confident the camera in its identification & classification.</p>
    </div>
""", unsafe_allow_html=True)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = loaded_model
        self.class_labels = class_labels

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Initial position and size of the green square
        square_size = 200

        # Get the dimensions of the frame
        frame_height, frame_width = img.shape[:2]

        # Calculate the initial position of the square to be centered
        x_start = frame_width // 2 - square_size // 2
        y_start = frame_height // 2 - square_size // 2

        # Draw the green square
        cv2.rectangle(img, (x_start, y_start), (x_start + square_size, y_start + square_size), (0, 255, 0), 2)

        # Extract the square region from the frame
        square_frame = img[y_start:y_start + square_size, x_start:x_start + square_size]

        # Resize the square region to match the input size expected by the model
        resized_frame = cv2.resize(square_frame, (200, 200))

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Prepare the image for prediction
        img_array = np.expand_dims(rgb_frame, axis=0)  # Add batch dimension
        img_array = img_array.astype('float32') / 255.0  # Normalize pixel values

        # Make predictions
        predictions = self.model.predict(img_array)

        # Get the predicted class label
        predicted_label = self.class_labels[np.argmax(predictions)]

        # Get the probability/confidence score for the predicted class
        confidence = np.max(predictions)  # Maximum confidence value

        # Display the predicted label
        text = f"Predicted: {predicted_label}"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display confidence below the predicted label
        confidence_text = f"Confidence: {confidence:.2f}"
        cv2.putText(img, confidence_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

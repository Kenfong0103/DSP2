import cv2
import numpy as np
from keras.models import model_from_json
import streamlit as st
from PIL import Image

# Error handling for imports
try:
    import cv2
    import numpy as np
    from keras.models import model_from_json
    import streamlit as st
    from PIL import Image
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

# Button to start the camera feed
st.text('Press "Start" to begin the camera.')
start_button = st.button('Start')

if start_button:
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

    # Start the camera
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        st.error("Could not open camera. Please check if camera is connected and accessible. Ensure no other application is using the camera, and allow permissions if prompted.")
    else:
        # Initial position and size of the green square
        square_size = 200

        stframe = st.empty()
        stop_button = st.button('Stop')
        
        while cap.isOpened():
            ret, frame = cap.read()  # Read a frame from the camera

            if not ret:
                st.error("Failed to capture image from camera.")
                break

            # Get the dimensions of the frame
            frame_height, frame_width = frame.shape[:2]

            # Calculate the initial position of the square to be centered
            x_start = frame_width // 2 - square_size // 2
            y_start = frame_height // 2 - square_size // 2

            # Clone the frame to draw on
            frame_clone = frame.copy()

            # Draw the green square
            cv2.rectangle(frame_clone, (x_start, y_start), (x_start + square_size, y_start + square_size), (0, 255, 0), 2)

            # Extract the square region from the frame
            square_frame = frame[y_start:y_start + square_size, x_start:x_start + square_size]

            # Resize the square region to match the input size expected by the model
            resized_frame = cv2.resize(square_frame, (200, 200))

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Prepare the image for prediction
            img_array = np.expand_dims(rgb_frame, axis=0)  # Add batch dimension
            img_array = img_array.astype('float32') / 255.0  # Normalize pixel values

            # Make predictions
            predictions = loaded_model.predict(img_array)

            # Get the predicted class label
            predicted_label = class_labels[np.argmax(predictions)]

            # Get the probability/confidence score for the predicted class
            confidence = np.max(predictions)  # Maximum confidence value

            # Display the predicted label
            text = f"Predicted: {predicted_label}"
            cv2.putText(frame_clone, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display confidence below the predicted label
            confidence_text = f"Confidence: {confidence:.2f}"
            cv2.putText(frame_clone, confidence_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert frame to Image for Streamlit
            img = Image.fromarray(cv2.cvtColor(frame_clone, cv2.COLOR_BGR2RGB))
            stframe.image(img, channels="RGB")

            # Check if the stop button is pressed
            if stop_button:
                break

        # Release the camera and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

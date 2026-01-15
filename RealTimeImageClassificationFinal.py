"""
Project Title : Real-Time Object Classification using CNN (MobileNetV2)
Description   : Performs real-time object classification using a pre-trained
                MobileNetV2 CNN model on webcam input.
Author        : Akash Subhash Guldagad
Date          : 17/12/2025
"""

# Import OpenCV for webcam access and image processing
import cv2

# Import NumPy for numerical operations
import numpy as np

# Import MobileNetV2 model and utility functions
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)


"""
Function Name : load_model
Description   : Loads the pre-trained MobileNetV2 CNN model
                with ImageNet weights.
Parameters    : None
Returns       : model (keras.Model) - Loaded CNN model
"""

def load_model():
    
    # Load MobileNetV2 with pre-trained ImageNet weights
    model = MobileNetV2(weights="imagenet")
    
    # Return the loaded model
    return model


"""
Function Name : preprocess_frame
Description   : Converts the input frame from BGR to RGB,
                resizes it to 224x224, and applies preprocessing
                required by MobileNetV2.
Parameters    :
               frame (numpy.ndarray) - Input frame captured from webcam
Returns       :
               x (numpy.ndarray) - Preprocessed frame ready for prediction
"""

def preprocess_frame(frame):
    
    # Convert frame color format from BGR (OpenCV) to RGB (CNN requirement)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize image to 224x224 as required by MobileNetV2
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Expand dimensions to match model input shape (batch size = 1)
    x = np.expand_dims(img_resized, axis=0).astype(np.float32)
    
    # Apply MobileNetV2 preprocessing (scaling & normalization)
    x = preprocess_input(x)
    
    # Return the preprocessed image
    return x


"""
Function Name : predict_object
Description   : Performs object classification on the preprocessed
                frame using the CNN model.
Parameters    :
    model (keras.Model)        - Loaded CNN model
    processed_frame (ndarray) - Preprocessed image
Returns       :
    label (str) - Predicted object label with confidence score
"""

def predict_object(model, processed_frame):
    
    # Perform prediction using the CNN model
    predictions = model.predict(processed_frame, verbose=0)
    
    # Decode the top prediction into a human-readable label
    decoded = decode_predictions(predictions, top=1)[0][0]
    
    # Format label with class name and confidence percentage
    label = f"{decoded[1]} : {decoded[2] * 100:.2f}%"
    
    # Return the formatted label
    return label


"""
Function Name : start_webcam_classification
Description   : Captures video from webcam, performs real-time
                object classification, and displays the results
                on the video feed.
Parameters    : None
Returns       : None
"""

def start_webcam_classification():
    
    # Load the CNN model
    model = load_model()
    
    # Initialize webcam capture (0 = default camera)
    cap = cv2.VideoCapture(0)

    print("Press 'q' to exit")

    # Infinite loop for real-time processing
    while True:
        
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        # Break loop if frame is not captured properly
        if not ret:
            break

        # Preprocess the captured frame
        processed_frame = preprocess_frame(frame)
        
        # Predict object label from the frame
        label = predict_object(model, processed_frame)

        # Display the prediction label on the video frame
        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        # Show the video output window
        cv2.imshow("Real-Time Object Classification", frame)

        # Exit loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam resources
    cap.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()


"""
Function Name : main
Description   : Entry point of the program.
Parameters    : None
Returns       : None
"""

def main():
    
    # Start the webcam-based object classification system
    start_webcam_classification()


# Execute main function only when script is run directly
if __name__ == "__main__":
    main()

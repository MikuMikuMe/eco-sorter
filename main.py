Certainly! Here’s a complete Python program for the "eco-sorter" project using computer vision. This program utilizes a machine learning model to classify waste materials, and it uses a webcam to capture images. The program requires you to have a pre-trained model capable of distinguishing between classes of waste (like paper, plastic, metal, etc.).

### Dependencies
To run this program, you'll need these Python libraries:
- OpenCV (`opencv-python`)
- TensorFlow or PyTorch for loading the machine learning model (e.g., `tensorflow` for TensorFlow)
- NumPy for image manipulation.

You can install these packages using pip:
```bash
pip install opencv-python tensorflow numpy
```

### Program

```python
import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to preprocess the image captured from webcam
def preprocess_image(image):
    try:
        # Resize the image to 224x224, the input size for the model
        image_resized = cv2.resize(image, (224, 224))
        # Normalize the image to [0, 1] range
        image_normalized = image_resized / 255.0
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        return image_batch
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Function to run prediction on the image and sort waste
def predict_waste(material_model, category_names, image):
    preprocessed_image = preprocess_image(image)
    if preprocessed_image is None:
        return "Processing Error"

    try:
        # Run prediction
        predictions = material_model(preprocessed_image)
        predicted_index = np.argmax(predictions[0])
        predicted_material = category_names[predicted_index]
        return predicted_material
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Prediction Error"

# Main function to capture image and classify
def main():
    # Path to the pre-trained model
    model_path = 'path_to_your_model.h5'
    
    # List of waste categories. These should correspond to your trained model's output categories.
    waste_categories = ['paper', 'plastic', 'metal', 'glass', 'organic', 'other']

    # Load the model
    model = load_model(model_path)
    if model is None:
        return

    # Open webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit.")    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            # Display the resulting frame
            cv2.imshow('Eco-Sorter', frame)

            # Wait for 's' key to capture image and classify
            if cv2.waitKey(1) & 0xFF == ord('s'):
                predicted_category = predict_waste(model, waste_categories, frame)
                print(f"Predicted Waste Category: {predicted_category}")

            # Exit the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release the capture and destroy all windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

### Notes
- **Model:** Replace `'path_to_your_model.h5'` with the path to your own pre-trained model.
- **Categories:** Ensure that `waste_categories` reflects the actual classes your model was trained on.
- **Camera Permissions:** Ensure your script has permission to access the webcam on your system.
- **Error Handling:** The code includes basic error handling to manage issues with model loading, camera access, and prediction errors.
  
This program captures live video from a webcam, processes each frame to make predictions on the type of waste, and displays the result. Adjust the model and categories as needed for your specific application.
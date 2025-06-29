import io
import os
import requests
from PIL import Image
from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from efficientnet.models import efficientnet_b0


MODEL_URL = "https://drive.google.com/file/d/1qgX-tJjrfNtz2AEb0FLO4SMdN9r33IC0" 
WEIGHTS_DIR = "weights"
MODEL_PATH = os.path.join(WEIGHTS_DIR, "best.ckpt")

if not os.path.exists(MODEL_PATH):
    print("Model not found, downloading...")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status() # Raise an exception for bad status codes
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded successfully.")
# --- Initialization ---
app = Flask(__name__)

NUM_CLASSES = 5 

# Load the trained model
try:
    state_dict_ema = torch.load('weights/best.ckpt', map_location='cpu', weights_only=False)
    
    model = efficientnet_b0(num_classes=NUM_CLASSES)
    
    model.load_state_dict(state_dict_ema['model'], strict=False)
    model.eval()
    print("PyTorch model loaded successfully.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model = None

# Define the image transformation pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(240),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load class names
try:
    with open("class_names.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(class_names)} class names.")
    # Verify that the number of class names matches the model's setting
    if len(class_names) != NUM_CLASSES:
        print(f"Warning: Number of classes in model ({NUM_CLASSES}) does not match number of names in class_names.txt ({len(class_names)})")
except Exception as e:
    print(f"Warning: Could not load class names. {e}")
    class_names = None

# --- Route to serve the HTML page ---
@app.route('/', methods=['GET'])
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload and returns the top prediction."""
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        top1_prob, top1_catid = torch.topk(probabilities, 1)
        
        # Add a check for safety
        predicted_index = top1_catid[0].item()
        if class_names and predicted_index < len(class_names):
            class_name = class_names[predicted_index]
        else:
            class_name = f"Class {predicted_index}" # Fallback
            
        confidence = top1_prob[0].item()
        result = {"class": class_name, "confidence": f"{confidence:.2%}"}
        return jsonify(result)
    except Exception as e:
        # This will print the actual error to your terminal for debugging
        print(f"Error during prediction: {e}") 
        return jsonify({'error': f'An error occurred on the server.'}), 500

# --- Main execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

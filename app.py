

from flask import Flask, render_template, request
import os
import torch
from werkzeug.utils import secure_filename
from mert_extractor import MertV1330MExtractor  # Your custom extractor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

input_dim = 1025  # Make sure this matches the embedding size from extractor

class CNNModel(nn.Module):
    def __init__(self, input_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Dynamically compute the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)  # (batch_size=1, channels=1, length=input_dim)
            out = self.pool2(F.relu(self.conv2(self.pool1(F.relu(self.conv1(dummy_input))))))
            flattened_dim = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flattened_dim, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, input_dim)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize model and load weights
model = CNNModel(input_dim=input_dim).to(device)
model.load_state_dict(torch.load('cnn_model2.pth', map_location=device))
model.eval()
print("Model loaded successfully.")

def predict_audio_embedding(embedding_vector):
    if not isinstance(embedding_vector, np.ndarray):
        raise ValueError("Input must be a NumPy array")
    print(f"Embedding vector shape: {embedding_vector.shape}")  # Debug print

    embedding_tensor = torch.tensor(embedding_vector, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(embedding_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()
        return predicted_class, confidence

# Initialize feature extractor
extractor = MertV1330MExtractor(device=device)
print("Feature extractor initialized.")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None

    if request.method == 'POST':
        if 'audio' not in request.files:
            return render_template('index.html', prediction="No file uploaded")

        file = request.files['audio']
        if file.filename == '':
            return render_template('index.html', prediction="No file selected")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(audio_path)

            try:
                features = extractor.extract_features(audio_path)
                if features is None:
                    raise ValueError("Failed to extract features")

                pred_class, confidence = predict_audio_embedding(features)

                # IMPORTANT: Adjust label logic here if your labels differ
                print(f" → Predicted Class: {'Real' if pred_class == 1 else 'Fake'} | Confidence: {confidence:.4f}")
                prediction = "Real Music 🎵" if pred_class == 1 else "Fake Music 🎭"

            except Exception as e:
                return render_template('index.html', prediction=f"Error: {str(e)}")

        else:
            return render_template('index.html', prediction="File type not allowed")

    return render_template('index.html', prediction=prediction, filename=filename)


if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)

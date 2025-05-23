Fake vs Real Music Detector
A web-based system to classify audio as real (human-composed) or fake (AI-generated) using pretrained MERT embeddings and a CNN mode

Features
Upload .wav audio file and detect if it’s AI-generated or real.

Uses m-a-p/MERT-v1-330M to extract audio embeddings.

Trained CNN model on the FakeMusicCaps dataset.

Interactive Flask web interface.

Installation
1. Clone the repository
2. Set up a virtual environment
3. Install dependencies
If you're on Windows and need local .whl files for PyTorch, install them like:
pip install torch-2.1.0+cpu-cp39-cp39-win_amd64.whl
pip install torchaudio-2.1.0+cpu-cp39-cp39-win_amd64.whl
pip install torchvision-0.16.0+cpu-cp39-cp39-win_amd64.whl
4. Running the Web App
python app.py

# Fake vs Real Music Detector

A **web-based system** to classify audio as **real (human-composed)** or **fake (AI-generated)** using **pretrained MERT embeddings** and a **CNN model**.
---

## Features

- 🎵 Upload `.wav` audio files and detect whether they are **AI-generated** or **human-composed**.
- 🔍 Uses [`m-a-p/MERT-v1-330M`](https://huggingface.co/m-a-p/MERT-v1-330M) to extract audio embeddings.
- 🧠 CNN model trained on the **FakeMusicCaps** dataset.
- 🌐 Interactive **Flask web interface**.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone [https://github.com/your-username/fake-vs-real-music-detector.git](https://github.com/Nutan22341/Music-DeepFake.git)
cd fake-vs-real-music-detector

2. Set up a virtual environment
3. Install dependencies
If you're on Windows and need local .whl files for PyTorch, install them like:
pip install torch-2.1.0+cpu-cp39-cp39-win_amd64.whl
pip install torchaudio-2.1.0+cpu-cp39-cp39-win_amd64.whl
pip install torchvision-0.16.0+cpu-cp39-cp39-win_amd64.whl
4. Running the Web App
python app.py

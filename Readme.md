
# 🎶 Fake vs Real Music Detector

A **web-based system** to classify audio as **real (human-composed)** or **fake (AI-generated)** using **pretrained MERT embeddings** and a **CNN model**.
---

## 🚀 Features

- 🎵 Upload `.wav` audio files and detect whether they are **AI-generated** or **human-composed**.
- 🔍 Uses [`m-a-p/MERT-v1-330M`](https://huggingface.co/m-a-p/MERT-v1-330M) to extract audio embeddings.
- 🧠 CNN model trained on the **FakeMusicCaps** dataset.
- 🌐 Interactive **Flask web interface**.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Nutan22341/Music-DeepFake.git
cd Music-DeepFake
```

### 2. Set up a virtual environment

```bash
python -m venv venv
```

Activate the environment:

- On **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- On **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you're on **Windows** and need local `.whl` files for **PyTorch**, install them like:

```bash
pip install torch-2.1.0+cpu-cp39-cp39-win_amd64.whl
pip install torchaudio-2.1.0+cpu-cp39-cp39-win_amd64.whl
pip install torchvision-0.16.0+cpu-cp39-cp39-win_amd64.whl
```

### 4. Run the Web App

```bash
python app.py
```

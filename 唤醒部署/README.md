# Wake Word Detection Script

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Customize for Your Model

Edit `wake_word_detector.py` and update these sections:

#### a. Model Loading (line ~45)
```python
def load_model(self, model_path):
    # Replace with your framework:
    
    # TensorFlow/Keras:
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)
    
    # PyTorch:
    # import torch
    # model = YourModelClass()
    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    # return model
```

#### b. Audio Preprocessing (line ~60)
```python
def preprocess_audio(self, audio_data):
    audio_array = np.array(audio_data, dtype=np.float32)
    audio_array = audio_array / 32768.0
    
    # Add your preprocessing:
    # - Extract MFCC features
    # - Convert to spectrogram
    # - Reshape to match your model input
    
    return audio_array
```

#### c. Model Inference (line ~80)
```python
def predict(self, audio_features):
    # TensorFlow/Keras:
    prediction = self.model.predict(audio_features, verbose=0)
    return float(prediction[0][0])
    
    # PyTorch:
    # with torch.no_grad():
    #     prediction = self.model(torch.FloatTensor(audio_features))
    # return float(prediction.item())
```

### 3. Configure Parameters (line ~145)
```python
MODEL_PATH = "path/to/your/model.h5"  # Your model file
SAMPLE_RATE = 16000  # Must match your training data
WINDOW_DURATION = 1.0  # Audio window length in seconds
THRESHOLD = 0.5  # Detection confidence threshold (0-1)
```

### 4. Run
```bash
python wake_word_detector.py
```

## Output

When wake word is detected, you'll see:
```
2024-02-05 10:30:45 - WARNING - 🎯 WAKE WORD DETECTED! (Confidence: 0.856)
```

## Troubleshooting

### PyAudio Installation Issues

**Mac:**
```bash
brew install portaudio
pip install pyaudio
```

**Linux:**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

### Adjust Audio Settings

If detection isn't working well, try adjusting:
- `SAMPLE_RATE`: Match your training data (usually 8000 or 16000)
- `WINDOW_DURATION`: Longer for longer wake words
- `THRESHOLD`: Lower for more sensitive detection, higher for fewer false positives

## Example Model-Specific Implementations

### TensorFlow with MFCC
```python
import librosa

def preprocess_audio(self, audio_data):
    audio_array = np.array(audio_data, dtype=np.float32) / 32768.0
    mfcc = librosa.feature.mfcc(y=audio_array, sr=self.sample_rate, n_mfcc=13)
    mfcc = mfcc.T  # Shape: (time_steps, 13)
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
    return mfcc
```

### PyTorch with Spectrogram
```python
import torch
import torchaudio

def preprocess_audio(self, audio_data):
    audio_tensor = torch.FloatTensor(audio_data) / 32768.0
    spectrogram = torchaudio.transforms.Spectrogram()(audio_tensor)
    spectrogram = spectrogram.unsqueeze(0)  # Add batch dimension
    return spectrogram
```

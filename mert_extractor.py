# mert_extractor.py
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor, AutoModel, AutoConfig
import torch.nn as nn

class MertV1330MExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        )
        config = AutoConfig.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        config.conv_pos_batch_norm = False
        self.model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", config=config, trust_remote_code=True).to(self.device)
        self.aggregator = nn.Conv1d(in_channels=25, out_channels=1, kernel_size=1).to(self.device)
        self.resample_rate = getattr(self.processor, "sampling_rate", 16000)

    # def extract_features(self, audio_path):
    #     # waveform, sampling_rate = torchaudio.load(audio_path)
    #     # if sampling_rate != self.resample_rate:
    #     #     waveform = T.Resample(orig_freq=sampling_rate, new_freq=self.resample_rate)(waveform)

    #     # inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=self.resample_rate, return_tensors="pt").to(self.device)

    #     # with torch.no_grad():
    #     #     outputs = self.model(**inputs, output_hidden_states=True)

    #     # all_hidden_states = torch.stack(outputs.hidden_states).squeeze(1)
    #     # time_reduced = all_hidden_states.mean(dim=1)
    #     # aggregated = self.aggregator(time_reduced.unsqueeze(0))
    #     # return aggregated.squeeze().detach().cpu().numpy()
    #     try:
    #         waveform, sampling_rate = torchaudio.load(audio_path)
    #     except Exception as e:
    #         print(f"Error loading {audio_path}: {e}")
    #         return None
        
    #     # Resample if needed
    #     if sampling_rate != self.resample_rate:
    #         waveform = T.Resample(orig_freq=sampling_rate, new_freq=self.resample_rate)(waveform)

    #     # Preprocess with processor
    #     inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=self.resample_rate, return_tensors="pt").to(self.device)

    #     with torch.no_grad():
    #         outputs = self.model(**inputs, output_hidden_states=True)

    #     # Hidden states: (25, batch=1, time, hidden_dim)
    #     all_hidden_states = torch.stack(outputs.hidden_states).squeeze(1)  # → (25, time, hidden_dim)
    #     time_reduced = all_hidden_states.mean(dim=1)  # → (25, hidden_dim)
    #     aggregated = self.aggregator(time_reduced.unsqueeze(0))  # → (1, 1, hidden_dim)

    #     return aggregated.squeeze().detach().cpu().numpy()
    def extract_features(self, audio_path):
        try:
            waveform, sampling_rate = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
        
        # Resample if needed
        if sampling_rate != self.resample_rate:
            waveform = T.Resample(orig_freq=sampling_rate, new_freq=self.resample_rate)(waveform)

        # Preprocess
        inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=self.resample_rate, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Extract hidden states → (25, time, hidden_dim)
        all_hidden_states = torch.stack(outputs.hidden_states).squeeze(1)
        time_reduced = all_hidden_states.mean(dim=1)                       # (25, hidden_dim)
        aggregated = self.aggregator(time_reduced.unsqueeze(0))           # (1, 1, hidden_dim)

        feature_vector = aggregated.squeeze().cpu()                       # (1024,)
        global_mean = feature_vector.mean().unsqueeze(0)                  # (1,)
        full_vector = torch.cat([feature_vector, global_mean])            # (1025,)
        
        print(f"[DEBUG] Extracted feature shape: {full_vector.shape}")    # (1025,)
        return full_vector.detach().cpu().numpy()
                                        # ✅ Final output


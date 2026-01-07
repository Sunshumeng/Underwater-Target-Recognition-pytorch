import librosa
import matplotlib.pyplot as plt
import numpy as np
import os

def load_audio(file_path):

    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def plot_waveform(y, sr, save_path=None):

    plt.figure(figsize=(12, 5))

    librosa.display.waveshow(y, sr=sr)

    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Amplitude', fontsize=12, fontweight='bold')

    plt.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    plt.show()

    plt.close()

def process_audio_files(folder_path):

    save_dir = os.path.join(folder_path, 'D:/音频分类/Pytorch-AudioClassification-master-main - MT/utils/wave')
    os.makedirs(save_dir, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            y, sr = load_audio(file_path)

            save_path = os.path.join(save_dir, os.path.splitext(filename)[0] + '_wave_plot.png')

            plot_waveform(y, sr, save_path)
            print(f"Saved waveform plot for {filename} to {save_path}")

if __name__ == "__main__":
    folder_path = "D:/音频分类/Pytorch-AudioClassification-master-main - MT/data/underwaterdata"
    process_audio_files(folder_path)

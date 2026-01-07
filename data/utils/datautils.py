import random
import warnings
import wave
import torch
import torchaudio
import librosa
import librosa.display

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

warnings.filterwarnings("ignore")

def audio_open(audio_path):
    """
    audio_path -> [tensor:channel*frames,int:sample_rate]
    """
    sig, sr = torchaudio.load(audio_path, channels_first=True)
    return [sig, sr]


def get_wave_plot(wave_path):
    f = wave.open(wave_path, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]

    str_bytes_data = f.readframes(nframes=nframes)
    wavedata = np.frombuffer(str_bytes_data, dtype=np.int16)
    wavedata = wavedata * 1.0 / (max(abs(wavedata)))
    time = np.arange(0, nframes) * (1.0 / framerate)

    plt.figure(figsize=(12, 5))  # 设置画布大小
    plt.plot(time, wavedata)

    # 添加英文坐标轴标签
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')  # x轴标签为"Time (s)"
    plt.ylabel('Normalized Amplitude', fontsize=12, fontweight='bold')  # y轴标签为"Normalized Amplitude"

    # 添加网格线以增强可读性
    plt.grid(alpha=0.3)

    # 调整布局以防止标签被裁剪
    plt.tight_layout()

    # 保存图像到指定路径
    wave_save_path = os.path.join("D:/音频分类/Pytorch-AudioClassification-master-main - MT/data/utils/wave",
                                  os.path.basename(wave_path).replace('.wav', '_wave_plot.png'))
    plt.savefig(wave_save_path, bbox_inches='tight', dpi=150)  # 提高分辨率

    # 关闭当前绘图窗口
    plt.close()



def regular_channels(audio, new_channels):
    """
    torchaudio-file([tensor,sample_rate])+target_channel -> new_tensor
    """
    sig, sr = audio
    if sig.shape[0] == new_channels:
        return audio
    if new_channels == 1:
        new_sig = sig[:1, :]
    else:
        new_sig = torch.cat([sig, sig], dim=0)
    return [new_sig, sr]

def regular_sample(audio, new_sr):
    sig, sr = audio
    if sr == new_sr:
        return audio
    channels = sig.shape[0]
    re_sig = torchaudio.transforms.Resample(sr, new_sr)(sig[:1, :])
    if channels > 1:
        re_after = torchaudio.transforms.Resample(sr, new_sr)(sig[1:, :])
        re_sig = torch.cat([re_sig, re_after])
    return [re_sig, new_sr]

def regular_time(audio, max_time):
    sig, sr = audio
    rows, len = sig.shape
    max_len = sr // 1000 * max_time

    if len > max_len:
        sig = sig[:, :max_len]
    elif len < max_len:
        pad_begin_len = random.randint(0, max_len - len)
        pad_end_len = max_len - len - pad_begin_len
        pad_begin = torch.zeros((rows, pad_begin_len))
        pad_end = torch.zeros((rows, pad_end_len))
        sig = torch.cat((pad_begin, sig, pad_end), 1)
    return [sig, sr]

def time_shift(audio, shift_limit):
    sig, sr = audio
    sig_len = sig.shape[1]
    shift_amount = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amount), sr)

# get Spectrogram


def extract_mel_spectrogram(file_path, sr=22050):
    # 强制加载1.5秒音频
    sig, sr = librosa.load(file_path, sr=sr, duration=1.5)  # 关键修改：固定时长

    # 统一频谱参数确保维度一致
    mel_spectrogram = librosa.feature.melspectrogram(y=sig, sr=sr,
                                                     n_mels=128,  # 固定mel带数
                                                     n_fft=1024,  # 固定帧长
                                                     hop_length=512)  # 固定跳跃长度
    log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel, sr  # 返回sr供保存谱图使用


def save_mel_spectrogram(mel_spectrogram, sr, file_path):
    """
    绘制并保存梅尔谱图
    """
    # 检查Mel频谱图中是否有nan或inf值
    if np.isnan(mel_spectrogram).any() or np.isinf(mel_spectrogram).any():
        raise ValueError("Mel spectrogram contains NaN or Inf values.")

    spectrogram_save_path = os.path.join("D:/音频分类/Pytorch-AudioClassification-master-main - MT/data/utils/Spectrogram1",
                                         os.path.basename(file_path).replace('.wav', '_mel_spectrogram.png'))
    print(f"即将保存梅尔谱图图像文件的路径为: {spectrogram_save_path}")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), sr=sr, x_axis='time', y_axis='mel',
                             cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time')
    plt.ylabel('Mel-bins')
    os.makedirs(os.path.dirname(spectrogram_save_path), exist_ok=True)
    plt.savefig(spectrogram_save_path, bbox_inches='tight')
    plt.close()


# 修改choice_and_process函数定义，添加默认参数
def choice_and_process(path, npy_dir=None, results=None):  # 修改1：添加npy_dir参数并设置默认值
    if results is None:
        results = []

        # 添加npy目录处理逻辑
    if npy_dir is not None:  # 修改2：当提供npy_dir时创建目录
        os.makedirs(npy_dir, exist_ok=True)

    if os.path.isdir(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isdir(file_path):
                results += choice_and_process(file_path, npy_dir)  # 修改3：传递npy_dir参数
            elif filename.endswith('.wav'):
                try:
                    mel_spectrogram, sr = extract_mel_spectrogram(file_path)
                    save_mel_spectrogram(mel_spectrogram, sr, file_path)

                    # 新增：保存npy文件
                    if npy_dir is not None:  # 修改4：当提供npy_dir时保存特征
                        npy_filename = os.path.splitext(filename)[0] + '.npy'
                        np.save(os.path.join(npy_dir, npy_filename), mel_spectrogram)

                    results.append(mel_spectrogram)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
        return results
    elif path.endswith('.wav'):
        try:
            mel_spectrogram, sr = extract_mel_spectrogram(path)
            save_mel_spectrogram(mel_spectrogram, sr, path)

            # 新增：保存npy文件
            if npy_dir is not None:
                npy_filename = os.path.splitext(os.path.basename(path))[0] + '.npy'
                np.save(os.path.join(npy_dir, npy_filename), mel_spectrogram)

            return [mel_spectrogram]
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            return []
    return results


# 修改调用方式（文件末尾部分）
folder_path = "D:/音频分类/Pytorch-AudioClassification-master-main - MT/data/underwaterdata1"
npy_dir = "D:/音频分类/Pytorch-AudioClassification-master-main - MT/data/scatter/npy_data1"  # 新增npy目录
result = choice_and_process(folder_path, npy_dir=npy_dir)  # 关键修改：传入npy_dir参数
print(f"返回结果类型: {type(result)}, 数量: {len(result)}")


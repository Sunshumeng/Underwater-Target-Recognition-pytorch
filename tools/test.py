import os
import matplotlib.pyplot as plt
import librosa
import numpy as np
import torch
import soundfile as sf
from matplotlib import pyplot as plt
from torchaudio.transforms import MelSpectrogram
from torch.nn.functional import normalize

from model.cnn import AudioClassificationModelCNN


def preprocess_audio(audio_path,
                     model_input_shape=(128, 344),
                     sample_rate=16000,
                     window_sec=0.15,  # 调大窗口保证n_fft
                     hop_sec=0.075):  # 保持50%重叠
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    # 调整静音切除阈值
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=30)

    # 动态计算帧参数（强制满足n_fft）
    frame_length = max(int(window_sec * sr), 2048)  # 保证最低2048点
    hop_length = int(hop_sec * sr)

    # 安全分帧
    frames = librosa.util.frame(
        trimmed_audio,
        frame_length=frame_length,
        hop_length=hop_length,
        axis=0
    )

    mel_specs = []
    # 修改为单层循环 + 使用enumerate获取索引
    for frame_idx, frame in enumerate(frames):  # 外层循环添加索引
        # 转换为PyTorch tensor
        waveform = torch.from_numpy(frame.copy()).float().squeeze(-1)

        # 梅尔频谱转换
        mel_transform = MelSpectrogram(
            sample_rate=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=model_input_shape[0]
        )
        mel_spec = mel_transform(waveform)

        # 修正位置：直接在此处调用绘图函数（缩进与mel_transform对齐）
        plot_spectrogram(mel_spec.numpy(), sr, index=frame_idx)  # 使用外层循环的索引

        # 后续处理保持原缩进
        mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = normalize(mel_spec)

        # 适配模型输入形状
        if mel_spec.shape[-1] < model_input_shape[1]:
            mel_spec = torch.nn.functional.pad(mel_spec, (0, model_input_shape[1] - mel_spec.shape[-1]))
        else:
            mel_spec = mel_spec[:, :, :model_input_shape[1]]

        # 通道适配 (根据模型第一层conv2d(2,8)的要求)
        mel_specs.append(mel_spec)  # Shape: [2,128,344]

    # 修改返回值
    return torch.stack(mel_specs).unsqueeze(1)  # [batch, channel, height, width]

# 新增函数（放在preprocess_audio函数下方）
def plot_spectrogram(spec, sr, save_dir="input_spectrograms", index=0):
    plt.figure(figsize=(10, 4))

    # 处理维度（移除通道维度）
    if spec.ndim == 3:  # [channels, n_mels, time]
        spec = spec.squeeze(0).mean(axis=0)  # 取单通道或平均多通道

    S_dB = librosa.power_to_db(spec, ref=np.max)

    # 显式设置坐标轴
    times = librosa.times_like(S_dB, sr=sr, hop_length=512)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)[:spec.shape[0]]

    librosa.display.specshow(S_dB,
                             x_coords=times,
                             y_coords=freqs,
                             x_axis='time',
                             y_axis='mel')

    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Original Mel Spectrogram - Frame {index}')
    plt.savefig(os.path.join(save_dir, f"frame_{index}.png"))
    plt.close()


def visualize_features(feat, layer_name):
        """
        可视化特征图
        :param feat: 特征图，形状为 [batch_size, channels, height, width]
        :param layer_name: 当前层的名称，用于标题
        """
        # 确保输入是4D张量
        if feat.ndim != 4:
            raise ValueError(f"特征图必须是4D张量，当前形状为 {feat.shape}")

        # 获取批次大小、通道数和特征图大小
        batch_size, channels, height, width = feat.shape

        # 仅可视化第一个样本的特征图
        feat = feat[0].detach().cpu()  # 去除梯度并移动到CPU

        # 设置子图布局
        fig, axes = plt.subplots(1, channels, figsize=(channels * 2, 2))
        if channels == 1:
            axes = [axes]  # 确保单个通道时也能正确处理

        # 绘制每个通道的特征图
        for i in range(channels):
            ax = axes[i]
            ax.imshow(feat[i], cmap='viridis')  # 使用颜色映射显示特征图
            ax.axis('off')  # 关闭坐标轴
            ax.set_title(f'Channel {i + 1}')

        # 设置总标题
        plt.suptitle(f'Feature Maps from {layer_name}', y=1.05)
        plt.tight_layout()
        plt.show()


def predict(model, input_tensor, model_checkpoint_path):
    """
    模型推理流程：
    1. 加载预训练权重
    2. 前向传播
    3. 分类结果处理
    """
    # 加载预训练权重
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)  # 忽略不匹配的键


    # 添加hook获取中间特征
    feature_maps = {}

    def get_features(name):
        def hook(model, input, output):
            feature_maps[name] = output.detach()

        return hook

    model.conv1.register_forward_hook(get_features('conv1'))
    model.conv2.register_forward_hook(get_features('conv2'))

    # 加载调整后的权重
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 前向传播后添加可视化
    with torch.no_grad():
        class_output, _ = model(query_x=input_tensor)

    # 新增特征可视化代码
    for layer_name, feat in feature_maps.items():
        visualize_features(feat, layer_name)

    # 前向传播
    with torch.no_grad():
        class_output, _ = model(query_x=input_tensor)

    # 转换为概率
    probabilities = torch.softmax(class_output, dim=1)
    return probabilities.numpy()


#  ---------------------------------------------------
if __name__ == "__main__":
    # 初始化模型
    model = AudioClassificationModelCNN(num_classes=32)  # 替换为实际类别数

    # 预处理音频
    audio_path = "D:/音频分类/Pytorch-AudioClassification-master-main - MT/data/underwaterdata/9-Fraser Dolphin-9100800F.wav"
    processed_input = preprocess_audio(audio_path)

    # 执行预测
    checkpoint_path = "D:/音频分类/Pytorch-AudioClassification-master-main - MT/workdir/exp-pytorch-audioclassification-master_2025_03_15_15_09/checkpoints/best_f1.pth"
    predictions = predict(model, processed_input, checkpoint_path)

    # 输出结果
    print(f"预测概率分布: {predictions}")
    print(f"预测类别: {np.argmax(predictions, axis=1)}")

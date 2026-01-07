import os
import re
import tqdm
import numpy as np
import pandas as pd
import librosa
import numpy as np
import torch

import librosa
import numpy as np
import tqdm
import os
import sys
import os
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 或若函数需自行实现，补充定义：
def extract_mel_spectrogram(filepath):
    # 实现梅尔频谱提取逻辑（需依赖librosa等库）
    import librosa
    y, sr = librosa.load(filepath)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    return mel_spec

# 获取项目根目录并添加到模块搜索路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.datautils import choice_and_process


class ScatterData:
    def __init__(self, csv_path, scatter_dir, classes_path, npy_dir):
        self.csv_path = csv_path
        self.scatter_dir = scatter_dir
        self.classes_path = classes_path
        self.npy_dir = npy_dir
        self.classes_list = self.getclasstxt()
        self.data = pd.DataFrame(columns=['filename', 'filepath', 'label', 'attribute'])

    def getclasstxt(self):
        # 改进：使用正则提取更复杂的文件名
        pattern = re.compile(r'\w+-\s*([\w\s]+)-\s*\w+\.wav')  # 匹配类别部分
        classes = set()
        for filename in os.listdir(self.scatter_dir):
            match = pattern.match(filename)
            if match:
                classes.add(match.group(1).strip())
        with open(self.classes_path, 'w') as f:
            f.write('\n'.join(sorted(classes)))
        return sorted(classes)  # 固定顺序

    def process(self):
        # 直接调用文档1的函数处理
        choice_and_process(self.scatter_dir, self.npy_dir)  # 传递npy保存路径

    def generator(self):
        data = []
        # 遍历npy目录
        for npy_file in os.listdir(self.npy_dir):
            if not npy_file.endswith('.npy'):
                continue

            # 解析文件名（示例文件名："0-Atlantic Spotted Dolphin-001.npy"）
            parts = npy_file.split('-')
            if len(parts) < 3:
                print(f"跳过无效文件名: {npy_file}")
                continue

            # 提取元数据
            attribute = parts[0].strip()
            class_name = parts[1].strip()
            label = self.classes_list.index(class_name)  # 获取类别索引

            # 构建记录
            record = {
                'filename': npy_file,
                'filepath': os.path.join(self.npy_dir, npy_file),
                'label': label,
                'attribute': attribute,
                'spec_shape': np.load(os.path.join(self.npy_dir, npy_file)).shape  # 记录特征维度
            }
            data.append(record)

        # 创建DataFrame并保存
        self.data = pd.DataFrame(data)
        # 检查特征维度一致性
        if self.data['spec_shape'].nunique() > 1:
            print(f"警告! 发现多种特征维度: {self.data['spec_shape'].unique()}")
        self.data.to_csv(self.csv_path, index=False)
        print(f"CSV生成完成，共{len(self.data)}条记录，保存至 {self.csv_path}")

# 实例化ScatterData类并调用相关方法
scatterdata = ScatterData(
    csv_path=r"D:/音频分类/Pytorch-AudioClassification-master-main - MT/data/scatter/refer1.csv",
    scatter_dir=r"D:/音频分类/Pytorch-AudioClassification-master-main - MT/data/underwaterdata1",
    classes_path=r"D:/音频分类/Pytorch-AudioClassification-master-main - MT/data/scatter/classes.txt",
    npy_dir=r"D:/音频分类/Pytorch-AudioClassification-master-main - MT/data/scatter/npy_data2",
)

scatterdata.process()
scatterdata.generator()





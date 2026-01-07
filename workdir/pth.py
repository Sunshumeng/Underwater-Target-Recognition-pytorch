import torch

checkpoint = torch.load('D:\音频分类\Pytorch-AudioClassification-master-main\workdir\exp-pytorch-audioclassification-master_2024_9_25_14_17\checkpoints/final_model.pth', map_location='cpu')

if isinstance(checkpoint, torch.nn.Module):
    checkpoint = checkpoint.state_dict()

keys = checkpoint.keys()
for key in keys:
    print(key)


checkpoint = torch.load('D:\音频分类\Pytorch-AudioClassification-master-main\workdir\exp-pytorch-audioclassification-master_2024_9_26_18_55\checkpoints/best_f1.pth', map_location='cpu')
print(checkpoint)
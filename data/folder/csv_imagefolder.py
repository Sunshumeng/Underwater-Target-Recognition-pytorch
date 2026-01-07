import os

from pandas import DataFrame

"""
A demo for generator DIF(datasets' information file) according to "ImageFolder Classification" format.
"""


def get_class_list(image_dir):
    return os.listdir(image_dir)


def generator(csv_path,image_dir):
    """
    data items:filename,filepath,label
    """
    class_list = get_class_list(image_dir)
    data = {'filename':[], 'filepath':[], 'label':[]}
    for class_name in class_list:
        class_path = image_dir + "/{}".format(class_name)
        for filename in os.listdir(class_path):
            filepath = class_path + '/{}'.format(filename)
            data['filename'].append(filename)
            data['filepath'].append(filepath)
            data['label'].append(class_list.index(class_name))
    dataframe = DataFrame(data=data)
    dataframe.to_csv(csv_path,encoding='utf-8')


csv_path = r'D:/音频分类/Pytorch-AudioClassification-master-main/data/folder/refer.csv'
image_dir = r'D:/音频分类/Pytorch-AudioClassification-master-main/data/音频数据'

generator(csv_path, image_dir)
# Pytorch-AudioClassification-master-main-MT Usage Guide

Project Structure
```
├─.idea
│  └─inspectionProfiles
├─data
│  ├─folder
│  ├─scatter
│  ├─utils
│  └─voc
├─datasets
│  └─__pycache__
├─model
│  └─__pycache__
├─tools
│  └─__pycache__
└─workdir


Process finished with exit code 0

```

## 1. Environment Setup


First, create the environment and install PyTorch. Activate the environment (refer to [this article](https://blog.csdn.net/Killer_kali/article/details/123173414?spm=1001.2014.3001.5501)):

```
conda create -n torchclassify python=3.7

```

Navigate to the project directory:

```
cd Pyotrch-AudioClassification-master-main-MT
```

Install required packages:

```
pip install -r requirements.txt
```

Modify the parameters under get_arg - the paths for the prediction audio feature file, weight file, and class information file. Then run the script.
## 2. Create and Train Your Own Data (Scatter Format)

Scatter format:
```
-datasets:
    -0.wav
    -1.wav
    .......
```


1. In the data directory, choose the corresponding format demo based on your dataset format;

2. Run the corresponding dataset information generation script to obtain DIF and other files;

3. Modify the parameters under the get_arg function in the train.py script;

4. Run the train.py script;

## 3.Other code coming soon
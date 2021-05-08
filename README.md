# Deep Learning - HW4: LeNet

Author: Jay Liao (re6094028@gs.ncku.edu.tw)

This is assignment 4 of Deep Learning, a course at Institute of Data Science, National Cheng Kung University. This project aims to construct LeNet-related models to perform image classification.

## Data

- Images: please go [here](https://drive.google.com/open?id=1kwYYWL67O0Dcbx3dvZIfbGg9NiHdyisr) to download raw image files and put them under the folder `./images/`. There are 64,225 files with 50 subfolders.

- File name lists of images: `./data/train.txt`, `./data/val.txt`, and `./data/test.txt`.

## Code

- `main_torch.py`: the main program for training LeNet-5 with PyTorch

- `main_keras.py`: the main program for training LeNet-5 with Keras

- Source codes for training LeNet-5 with PyTorch:

    -  `./lenet_torch/args.py`: define the arguments parser

    -  `./lenet_torch/models.py`: construct the models
    
    -  `./lenet_torch/trainer.py`: class for training, predicting, and evaluating the models
    
    -  `./lenet_torch/utils.py`: little tools

- Source codes for training LeNet-5 with Keras:

    -  `./lenet_keras/args.py` defines the arguments parser
    
    -  `./lenet_keras/trainer.py`: class for training, predicting, and evaluating the models
    
    -  `./lenet_keras/utils.py`: little tools

## Folders

- `./images/` should contain raw image files (please go [here](https://drive.google.com/open?id=1kwYYWL67O0Dcbx3dvZIfbGg9NiHdyisr) to download and put them with subfolders here).

- `./data/` contains .txt files of image lists.

- `./output_torch/` and `./output_keras/` will contain trained models, model performances, and experiments results after running. 

## Requirements

```
numpy==1.16.3
pandas==0.24.2
tqdm==4.50.0
opencv-python==3.4.2.16
matplotlib==3.1.3
torch==1.7.1
keras==2.4.3
tensorflow==2.3.1
tensorflow-gpu==2.3.1
```

## Usage

1. Clone this repo.

```
git clone https://github.com/jayenliao/DL-LeNet.git
```

2. Set up the required packages.

```
cd DL-LeNet
pip3 install requirement.txt
```

3. Run the experiments.

```
python3 main_torch.py
python3 main_keras.py
```

## Reference

1. Liao, J. C. (2021). Deep Learning - Image Classification. GitHub: https://github.com/jayenliao/DL-image-classification.

2. Liao, J. C. (2021). Deep Learning - Computational Graph. GitHub: https://github.com/jayenliao/DL-computational-graph.

3. Lowe, D. G. (1999, September). Object recognition from local scale-invariant features. In Proceedings of the seventh IEEE international conference on computer vision (Vol. 2, pp. 1150-1157). Ieee.

4. Bay, H., Ess, A., Tuytelaars, T., & Van Gool, L. (2008). Speeded-up robust features (SURF). Computer vision and image understanding, 110(3), 346-359.

5. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

6. 斎藤康毅（吳嘉芳譯）（2017）。Deep Learning: 用Python進行深度學習的基礎理論實作。碁峰資訊股份有限公司。ISBN: 9789864764846。GitHub: https://github.com/oreilly-japan/deep-learning-from-scratch。

7. Watt, J., Borhani, R., & Katsaggelos, A. K. (2019). Machine learning refined. ISBN: 9781107123526. GitHub: https://github.com/jermwatt/machine_learning_refined.
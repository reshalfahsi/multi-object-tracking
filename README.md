# Multi-Object Tracking Using FCOS + DeepSORT


<div align="center">
    <a href="https://colab.research.google.com/github/reshalfahsi/multi-object-tracking/blob/master/Multi_Object_Tracking.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a>
    <br />
</div>



<div align="center">
    <img src="https://github.com/reshalfahsi/multi-object-tracking/blob/master/assets/KITTI-19.gif" alt="KITTI-19"></a>
    <br />
</div>



> An idiot admires complexity, a genius admires simplicity... 
> 
> ― Terry Davis



As the term suggests, multi-object tracking's primary pursuit in computer vision problems is tracking numerous detected objects throughout a sequence of frames. This means multi-object tracking embroils two subproblems, i.e., detection and tracking. In this project, the object detection problem is tackled via COCO dataset-pretrained FCOS, an anchor-free proposal-free single-stage object detection architecture. Meanwhile, the tracking problem is solved through the DeepSORT algorithm. To track each object, DeepSORT utilizes the Kalman filter and the re-identification model. The Kalman filter is widely used to predict the states of a certain system. In this case, the states are the pixel positions (``cx``, ``cy``), the bounding box's aspect ratio and height (``w/h``, ``h``), and the velocity of ``cx``, ``cy``, ``w/h``, and ``h`` of the objects. This project makes use of the simplified DeepSORT algorithm. The re-identification model aids in pinpointing two identical objects between frames based on their appearance. This model generates a vector descriptor associated with the objects in a frame. ImageNet-1K dataset-pretrained MobileNetV3-Small is leveraged as the backbone of the re-identification model. FAISS is set on duty in the matching process of an object's appearance and pixel location in consecutive frames. Here, the datasets used for fine-tuning the re-identification model and evaluating the tracking are Market-1501 and MOT15, respectively. The train set of the MOT15 dataset is used for testing (producing the quantitative result) and the test set of the MOT15 dataset is used for inferencing (producing the qualitative result). This project sets the object to be tracked is the person.


## Experiment

Proceed to this [notebook](https://github.com/reshalfahsi/multi-object-tracking/blob/master/Multi_Object_Tracking.ipynb) to scrutinize the re-identification and tracking.


## Result

### Re-identification

#### Quantitative Result

Quantitatively speaking, the loss and accuracy of the re-identification model on the test set are revealed in this table:

Test Metric  | Score
------------ | -------------
Loss         | 0.258
Accuracy     | 95.69%


#### Accuracy and Loss Curve

<p align="center"> <img src="https://github.com/reshalfahsi/multi-object-tracking/blob/master/assets/loss_curve.png" alt="loss_curve" > <br /> The loss curve on the train and validation sets of the re-identification model. </p>

<p align="center"> <img src="https://github.com/reshalfahsi/multi-object-tracking/blob/master/assets/acc_curve.png" alt="acc_curve" > <br /> The accuracy curve on the train and validation sets of the re-identification model. </p>


#### Qualitative Result

The qualitative result that shows several samples of distinguishable instances and their look-alike is conveyed through these collated images:

<p align="center"> <img src="https://github.com/reshalfahsi/multi-object-tracking/blob/master/assets/qualitative_reid.png" alt="qualitative" > <br /> Some instances and their other similar instances of the Market-1501 dataset. </p>


### Tracking

#### Quantitative Result

Here, the quantitative result of the tracking algorithm:

```
                IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT  ML    FP    FN IDs   FM   MOTA  MOTP IDt IDa IDm
PETS09-S2L1    56.6% 55.3% 57.8% 89.1% 85.2%  19  14   5   0   693   489  36  127  72.8% 0.238   6  29   4
KITTI-13       42.3% 36.0% 51.2% 52.8% 37.1%  42   7  25  10   681   360   8   22 -37.7% 0.259   2   7   1
KITTI-17       70.1% 67.4% 73.1% 79.6% 73.5%   9   6   3   0   196   139  10   18  49.5% 0.228   4   5   0
ADL-Rundle-8   42.8% 41.6% 44.1% 66.4% 62.7%  28  14  12   2  2681  2280  74  194  25.8% 0.269  11  61   2
ADL-Rundle-6   52.1% 56.4% 48.4% 62.2% 72.5%  24   6  16   2  1181  1891  62   94  37.4% 0.218   8  51   1
ETH-Sunnyday   68.1% 58.1% 82.3% 85.8% 60.5%  30  14  15   1  1040   263  12   46  29.2% 0.200   1  11   1
TUD-Campus     55.6% 48.2% 65.7% 70.8% 51.8%   8   3   5   0   236   105   7   14   3.1% 0.240   1   5   0
Venice-2       48.0% 44.7% 51.8% 69.6% 59.9%  26  10  16   0  3320  2172  64  140  22.2% 0.234   4  56   1
ETH-Pedcross2  53.1% 67.2% 43.9% 48.9% 74.7% 133  14  62  57  1034  3203  97  157  30.8% 0.240  22  84  13
ETH-Bahnhof    48.8% 40.8% 60.8% 71.7% 48.1% 171  74  67  30  4195  1533  56  133  -6.8% 0.236  45  42  38
TUD-Stadtmitte 75.3% 81.2% 70.2% 80.1% 92.6%  10   6   4   0    74   230  13   18  72.6% 0.224   1  12   0
OVERALL        51.5% 49.8% 53.2% 68.3% 64.0% 500 168 230 102 15331 12665 439  963  28.7% 0.237 105 363  61
```


#### Qualitative Result

These are the visualizations of the multi-object tracking on the 10-second video clips of the test set of MOT15.

<p align="center"> <img src="https://github.com/reshalfahsi/multi-object-tracking/blob/master/assets/ADL-Rundle-1.gif" alt="ADL-Rundle-1" > <br /> The result on ADL-Rundle-1. </p>
<p align="center"> <img src="https://github.com/reshalfahsi/multi-object-tracking/blob/master/assets/ADL-Rundle-3.gif" alt="ADL-Rundle-3" > <br /> The result on ADL-Rundle-3. </p>
<p align="center"> <img src="https://github.com/reshalfahsi/multi-object-tracking/blob/master/assets/AVG-TownCentre.gif" alt="AVG-TownCentre" > <br /> The result on AVG-TownCentre. </p>
<p align="center"> <img src="https://github.com/reshalfahsi/multi-object-tracking/blob/master/assets/ETH-Crossing.gif" alt="ETH-Crossing" > <br /> The result on ETH-Crossing. </p>
<p align="center"> <img src="https://github.com/reshalfahsi/multi-object-tracking/blob/master/assets/ETH-Jelmoli.gif" alt="ETH-Jelmoli" > <br /> The result on ETH-Jelmoli. </p>
<p align="center"> <img src="https://github.com/reshalfahsi/multi-object-tracking/blob/master/assets/ETH-Linthescher.gif" alt="ETH-Linthescher" > <br /> The result on ETH-Linthescher. </p>
<p align="center"> <img src="https://github.com/reshalfahsi/multi-object-tracking/blob/master/assets/KITTI-16.gif" alt="KITTI-16" > <br /> The result on KITTI-16. </p>
<p align="center"> <img src="https://github.com/reshalfahsi/multi-object-tracking/blob/master/assets/KITTI-19.gif" alt="KITTI-19" > <br /> The result on KITTI-19. </p>
<p align="center"> <img src="https://github.com/reshalfahsi/multi-object-tracking/blob/master/assets/PETS09-S2L2.gif" alt="PETS09-S2L2" > <br /> The result on PETS09-S2L2. </p>
<p align="center"> <img src="https://github.com/reshalfahsi/multi-object-tracking/blob/master/assets/TUD-Crossing.gif" alt="TUD-Crossing" > <br /> The result on TUD-Crossing. </p>
<p align="center"> <img src="https://github.com/reshalfahsi/multi-object-tracking/blob/master/assets/Venice-1.gif" alt="Venice-1" > <br /> The result on Venice-1. </p>


## Credit

- [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/pdf/1904.01355.pdf)
- [Simple Online and Realtime Tracking With a Deep Association Metric](https://arxiv.org/pdf/1703.07402.pdf)
- [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)
- [Market-1501 Dataset](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
- [MOT15](https://motchallenge.net/data/MOT15/)
- [MOTChallenge MOT17 Data View](https://www.kaggle.com/code/stpeteishii/motchallenge-mot17-data-view)
- [MOT15-D2-SORT-DEEPSORT](https://colab.research.google.com/drive/1hhtoMwFxpOGXiXtIjEBeST9rQ6BdF5Gm)
- [Deep SORT](https://github.com/levan92/deep_sort_realtime)
- [py-motmetrics](https://github.com/cheind/py-motmetrics)
- [Faiss: A library for efficient similarity search](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
- [Introduction to Facebook AI Similarity Search (Faiss)](https://www.pinecone.io/learn/series/faiss/faiss-tutorial/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)

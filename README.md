# Deep Rotation Correction without Angle Prior ([paper](https://arxiv.org/abs/2207.03054))
<p align="center">Lang Nie<sup>1</sup>, Chunyu Lin<sup>1 *</sup>, Kang Liao<sup>1</sup>, Shuaicheng Liu<sup>2</sup>, Yao Zhao<sup>1</sup></p>
<p align="center"><sup>1</sup>Beijing Jiaotong University</p>
<p align="center"><sup>2</sup>University of Electronic Science and Technology of China</p>
<p align="center"><sup>{nielang, cylin, kang_liao, yzhao}@bjtu.edu.cn, liushuaicheng@uestc.edu.cn</sup></p>

<div align=center>
<img src="https://github.com/nie-lang/RotationCorrection/blob/main/fig1.jpg"/>
</div>
<p align="center"><sup>Fig.1. Different solutions to correct the tilted image. Our solution (e) can eliminate the tilt without angle prior, while the others (b)(c)(d)(f)(g)(h) require an accurate rotated angle. The red square denotes the cropping region, and the arrow highlights the distorted area. The horizontal and vertical dotted lines are drawn to help observe the slight tilt.</sup></p>

## Dataset (DRC-D)
We build this dataset by He et al.'s content-aware rotation and further manual correction as follows:
<div align=center>
<img src="https://github.com/nie-lang/RotationCorrection/blob/main/dataset.jpg"/>
</div>
<p align="center"><sup>Fig.2. The process of dataset generation. We further correct the randomly rotated result generated from He et al.’ rotation. The red arrows in (c) indicate the manual adjustment of moving the mesh vertices. He et al.’s rotation neglects the rotation of the cross ((b) right), while our manual correction slightly rotates it to produce a more natural appearance ((c) right).</sup></p>

Every example includes three items: a input image (a tilted image), a tilted angle and a label (a corrected image). For simplicity, we leverage the name of the input image to record the tilted angle, e.g., "00014_-7.1.jpg" indicates the input image has a tilt of -7.1°.

Now, the dataset can be downloaded in in [Google Drive](https://drive.google.com/drive/folders/1y8964QKakL1zJsuzuivCx41_YkrsOKv_?usp=share_link) or [Baidu Cloud](https://pan.baidu.com/s/1WByNz64oNoSRbuzCgcnXGQ)(Extraction code: 1234).

## Requirement
* python 3.6
* numpy 1.18.1
* tensorflow 1.13.1

More details about the environment can be found [here](https://github.com/nie-lang/DeepRectangling/issues/4).

## Training
#### Step 1: Download the pretrained vgg19 model
Download [VGG-19](https://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models). Search imagenet-vgg-verydeep-19 in this page and download imagenet-vgg-verydeep-19.mat. Then please place it to 'Codes/vgg19/' folder.

#### Step 2: Train the network
Modify the 'Codes/constant.py' to set the 'TRAIN_FOLDER'/'ITERATIONS'/'GPU'. In our experiment, we set 'ITERATIONS' to 150,000. 

```
python train.py
```

## Testing
#### Pretrained model for deep rotation correction
Our pretrained rectangling model can be available at [Google Drive](https://drive.google.com/drive/folders/1gEsE-7QBPcbH-kfHqYYR67C-va7vztxO?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/19jRzz_1E97X35j6qmWm_kg)(Extraction code: 1234). And place the four files to 'Codes/checkpoints/pretrained_model/' folder.
#### Testing 
Modidy the 'Codes/constant.py'to set the 'TEST_FOLDER'/'GPU'. The path for the checkpoint file can be modified in 'Codes/inference.py'.

```
python inference.py
```
#### Testing with your own data
We have specified the path for other datasets in 'Codes/constant.py'. You can collect your own tilted images and place it to 'Other_dataset/input/'.

```
python inference2.py
```
The corrected results can be found in 'Other_dataset/correction/'.

## Citation
```
@article{nie2022deep,
  title={Deep Rotation Correction without Angle Prior},
  author={Nie, Lang and Lin, Chunyu and Liao, Kang and Liu, Shuaicheng and Zhao, Yao},
  journal={arXiv preprint arXiv:2207.03054},
  year={2022}
}
```

## References
[1] Nie et al., “Depth-Aware Multi-Gird Deep Homogrpahy Estimation with Contextual Correlation,” TCSVT, 2021.  
[2] Nie et al., “Deep Rectangling for Image Stitching: A Learning Baseline,” CVPR, 2022.  

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

## Code
Coming soon. 

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

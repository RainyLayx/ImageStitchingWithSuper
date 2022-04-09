# ImageStitchingWithSuper
## Introduction
An longitudinal image stitching solution with SuperPoint(https://arxiv.org/abs/1712.07629) and SuperGlue(https://arxiv.org/abs/1911.11763)

## Contents
`demo_superglue.py` : runs a live demo on a webcam, IP camera, image directory or movie file

`match_pairs.py`: reads image pairs from files and dumps matches to disk (also runs evaluation if ground truth relative poses are provided)
* When you input 2 pieces of a image,the program will firstly extract the features of them by SuperPoint.If you input the keypoints manually,it will directly match them,but I think you have no need to do this.

`npzread.py`: analyze the output of match_pairs.py

`GenerateAnnos.py`: generate groups of image name in test directories.
* The words before the penultimate 6th of images in one group should be the same

`ImageStiWithSuper.py`: stitch the image with 2 pieces

`ImageStiWithSuper_multi.py`: stitch the image with more than 2 pieces

## Dependencies
* Python 3 == 3.6.4
* PyTorch == 1.9.0
* OpenCV == 3.4.2 
* Matplotlib == 3.2.2
* NumPy == 1.19.5

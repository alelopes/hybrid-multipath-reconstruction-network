# Hybrid Multipath Reconstruction Network

This is our implementation for the M-L Unicamp Team submission team in the MC-MRRec challenge, available at https://sites.google.com/view/calgary-campinas-dataset/mr-reconstruction-challenge?authuser=0

Although hybrid reconstruction methods have already been proposed in the literature [1][2], most of them use sequential models with stages connected in a single path. We propose a method that uses several networks in parallel, creating stages with individual objectives. Consequently, our method is composed of seven modules divided into sequential and parallel stages in cascade. Each module is represented by a U-net [3] based architecture.

The method is composed of three main stages, as in Fig 1.


<img src="https://github.com/alelopes/hybrid-multipath-reconstruction-network/blob/main/imgs/img1.png" width="500">
Figure 1: Fluxogram of the proposed method. Highlighted in green are the main stages of the method. Yellow boxes highlight the blocks for the first stage. Blue boxes represent the modules. K is for K-space and I for image domain.

# Code

The code is highly inspired by https://github.com/rmsouza01/CD-Deep-Cascade-MR-Reconstruction and you can consider this project as a fork of the original project. We changed some folder names and added a new file in the src folder (originally Module). Also, our notebook is a modification of the multi-coil training notebook available in the same repository.

All codes for reproducing training results are available in [training notebook](https://github.com/alelopes/hybrid-multipath-reconstruction-network/blob/main/notebooks/train_mc.ipynb)

# Team

Team is composed by:

Livia Rodrigues - MicLab - FEEC/Unicamp
Alexandre Lopes - Institute of Computing - Unicamp
Helio Pedrini - Institute of Computing - Unicamp
Leticia Rittner - MicLab - FEEC/Unicamp

# References

[1]  Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutionalnetworks for biomedical image segmentation.CoRR, abs/1505.04597, 2015.[2]  Roberto Souza, R Marc Lebel, and Richard Frayne. A hybrid, dual domain,cascade of convolutional neural networks for magnetic resonance image re-construction.   InInternational  Conference  on  Medical  Imaging  with  DeepLearning, pages 437–446. PMLR, 2019.[3]  Shanshan  Wang,  Ziwen  Ke,  Huitao  Cheng,  Sen  Jia,  Leslie  Ying,  HairongZheng,  and  Dong  Liang.   Dimension:  Dynamic  mr  imaging  with  both  k-space  and  spatial  prior  knowledge  obtained  via  multi-supervised  networktraining.NMR in Biomedicine, page e4131, 2019

[2]  Roberto Souza, R Marc Lebel, and Richard Frayne. A hybrid, dual domain,cascade of convolutional neural networks for magnetic resonance image re-construction.   InInternational  Conference  on  Medical  Imaging  with  DeepLearning, pages 437–446. PMLR, 2019

[3]  Shanshan  Wang,  Ziwen  Ke,  Huitao  Cheng,  Sen  Jia,  Leslie  Ying,  HairongZheng,  and  Dong  Liang.   Dimension:  Dynamic  mr  imaging  with  both  k-space  and  spatial  prior  knowledge  obtained  via  multi-supervised  networktraining.NMR in Biomedicine, page e4131, 2019

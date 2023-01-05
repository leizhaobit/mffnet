# mffnet
 A deepfake detection network fusing RGB features and textural information extracted by neural networks and signal processing methods


 Mffnet consists of four key components: (1) a feature extraction module to further extract textural
and frequency information using the Gabor convolution and residual attention blocks; (2) a texture
enhancement module to zoom into the subtle textural features in shallow layers; (3) an attention
module to force the classifier to focus on the forged part; (4) two instances of feature fusion to firstly
fuse textural features from the shallow RGB branch and feature extraction module and then to fuse
the textural features and semantic information. 

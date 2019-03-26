## multi-module CNN models

These scripts generate results in our manuscript entitled "Unraveling of transcription factor-DNA binding mechanisms through interpretable multi-module deep learning model". In this work, we built CNN models to study transcription factor (TF)-DNA binding both in vitro and in vivo. After well-performed models were trained, we interpreted binding mechanisms through network interpretation methods. The workflow is as follows: 

<img src="https://github.com/xinbeibei/multi-module_CNN/blob/master/Picture1.png" width=800 />

After collecting SELEX-seq data for eight Exd-Hox heterodimers in Drosophila, we systematically evaluated the strategies to handle DNA sequence orientations in sequence-based CNN models, they are CNN (RC model), CNN (canonical), CNN (canonical, RC augmented), and CNN (canonical, double sample) models. 

<img src="https://github.com/xinbeibei/multi-module_CNN/blob/master/Picture3A.png" width=300 />

Based on well-trained models, we evaluated four interpretation methods: Gradient*input, DeconvNet, DeepLIFT, and in silico mutagenesis (ISM). We found that reverse-complement weight sharing CNN models, together with ISM interpretation mehotd, are robust and accurate approaches to model binding specificity of eight Exd-Hox heterodimers and validate their in vivo binding events on the *svb* enhancer. 

## Dependencies

The pipeline requires:

* python 2.7
* DeepLIFT (citation) version at https://github.com/kundajelab/deeplift/tree/v0.6.6.2-alpha
* seq2logo
* Weblogo2



## Tutorial


## Project home page

For information on teh source tree, examples, isuses, and pull requests, see 

	https://github.com/xinbeibei/multi-module_CNN


## multi-module CNN models

These scripts generate results in our manuscript entitled "Unraveling of transcription factor-DNA binding mechanisms through interpretable multi-module deep learning model". In this work, we built CNN models to study transcription factor (TF)-DNA binding both in vitro and in vivo. After well-performed models were trained, we interpreted binding mechanisms through network interpretation methods. The workflow is as follows: 

<img src="https://github.com/xinbeibei/multi-module_CNN/blob/master/Picture1.png" width=800 />

After collecting SELEX-seq data for eight Exd-Hox heterodimers in Drosophila, we systematically evaluated the strategies to handle DNA sequence orientations in sequence-based CNN models, they are CNN (RC model), CNN (canonical), CNN (canonical, RC augmented), and CNN (canonical, double sample) models. 

<img src="https://github.com/xinbeibei/multi-module_CNN/blob/master/Picture3A.png" width=300 />

Based on well-trained models, we evaluated four interpretation methods: Gradient*input, DeconvNet, DeepLIFT, and in silico mutagenesis (ISM). We found that reverse-complement weight sharing CNN models, together with ISM interpretation mehotd, are robust and accurate approaches to model binding specificity of eight Exd-Hox heterodimers and validate their in vivo binding events on the Svb enhancer. 

## Dependencies

The pipeline requires:

* python 2.7
* DeepLIFT (citation) version at https://github.com/kundajelab/deeplift/tree/v0.6.6.2-alpha
* seq2logo
* Weblogo2

## Tutorial

1. Build L2-regularized multiple linear regression (L2-MLR) model with aligned SELEX-seq data. Take Exd-Scr for an example. 

```sh
mkdir ../data/SELEX_aligned
cd codes_MLR
R --no-restore --no-save --args ../data/SELEX_aligned/Scr/Scr.txt ../data/SELEX_aligned/Scr/encoded 8 feature_list < encode_custom.R 2>&1 1>/dev/null 
R --no-restore --no-save --args Scr ../data/SELEX_aligned/Scr/input 10 feature_list < shuffle_divide.R 2>&1 1>/dev/null
python train.py Scr ../data/SELEX_aligned/Scr/encoded ../data/SELEX_aligned/Scr/input ../data/SELEX_aligned/Scr/output 10 train.R feature_list
python test.py Scr ../data/SELEX_aligned/Scr/encoded ../data/SELEX_aligned/Scr/input ../data/SELEX_aligned/Scr/output 10 predict.R feature_list
python summarize.py Scr ../data/SELEX_aligned/Scr/output summarize.R ../data/SELEX_aligned/Scr/result feature_list
```

1. Build four sequence-based CNN models on SELEX-seq data. Here we took CNN (RC model) for an example. After running the following command, the top 3 hyperparameter settings would be selected and their corresponding CNN models and training/testing performance would be saved. Note that the prior generated hyperparameter list is in lr_file. 

```sh
cd codes
# model options are canonical/RCaugmented/RCmodel/double.
python train_CNN_SELEX.py RCmodel --steps train,test --tfs Scr,Ubx,Antp,Pb,Dfd,AbdA,AbdB,Lab --lr_file lr_file 
```

2. Interpret well-trained models with Gradient*input, DeconvNet, and ISM. For example, we could use following commands to obtain unit-resolution importance matrix of ATGATTTATTACCC (high-affinity BS of Exd-Ubx) interpreted by CNN (RC model) of Exd-Ubx.
 
```sh
python interpret_CNN_SELEX.py RCmodel ISM --interpret_file ../data/interpret_seq/ATGATTTATTACCC.h5 --tfs Ubx --lr_file ../out/SELEX_RCmodel/Ubx/train_perf.txt
```

Note that the .h5 file in --interpret_file only contains one sequence for interpretation. For ISM method, the sequence to be interpret could be a long sequence, for example, Svb enhancer as prepared in ../data/long_low_affinity_WT_short_seqs.h5. 

The fourth interpretation method DeepLIFT was implemented separately.

3. Free energy change of different mutations on the Svb enhancer. 

```sh
for seq in WT mut1 mut2 mut3 mut12 site1 site2 site3A site3B site3C site3D site3E
do
	for tf in Scr Ubx
	do
 		head -n1 ../out/SELEX_RCmodel/$tf/train_perf.txt > ../out/SELEX_RCmodel/$tf/best_perf.txt
 		python train_CNN_SELEX.py RCmodel --steps predict --tfs $tf --pred_file ../data/interpret_seq/long_low_affinity_$seq\_short_seqs.h5 --lr_file ../out/SELEX_RCmodel/$tf/best_perf.txt
 		python train_CNN_SELEX.py RCmodel --steps deltadeltadeltaG --tfs $tf --mut_str $seq 
done
done 
```

4. After training a CNN model, align training data with one certain motif scanner, and create either a position weight matrix or YR logo. Here we use Exd-Ubx and CNN (canonical) model as an example. 

```sh
#First step is to find out which motif scanner has largest information content.
python train_CNN_SELEX.py canonical --steps align --tfs Ubx
matrix-clustering -v 1 -max_matrices 300 -matrix ubx filters_0,0.001,0.0003.meme meme -hclust_method average -calc sum -title 'ubx' -metric_build_tree 'Ncor' -lth w 5 -lth cor 0.6 -lth Ncor 0.4 -quick -label_in_tree name -return json,heatmap -o ./matrix-clustering 2> ./matrix-clustering_err.txt

sort -nk7 ./matrix-clustering_tables/pairwise_compa_matrix_descriptions.tab > test

#second step: set align_to_one_filter=True, and get aligned sequences based on one filter
python train_CNN_SELEX.py canonical --steps align --tfs Ubx
sort -nk2 aligned_0,0.001,0.0003_filter3.txt > aligned_0,0.001,0.0003_filter3_sorted.txt
```

## Project home page

For information on teh source tree, examples, isuses, and pull requests, see 

	https://github.com/xinbeibei/multi-module_CNN


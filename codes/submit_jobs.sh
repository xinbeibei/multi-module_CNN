#!/bin/sh

for seq in WT mut1 mut2 mut3 mut12 site1 site2 site3A site3B site3C site3D site3E
do
	cp /home/bxin/Documents/low_affinity/viz_data/long_low_affinity_$seq\_short_seqs.h5 ../data/interpret_seq/
done

############   train and test models   ##################
# python train_CNN_SELEX.py RCmodel --steps train,test --tfs Scr,Ubx,Antp,Pb --lr_file lr_file 
# python train_CNN_SELEX.py RCmodel --steps train,test --tfs Dfd,AbdA,AbdB,Lab --lr_file lr_file 

# python simple_train_CNN_SELEX.py RCmodel --steps train,test --tfs Scr,Ubx,Antp,Pb --lr_file lr_file 
# python simple_train_CNN_SELEX.py RCmodel --steps train,test --tfs Dfd,AbdA,AbdB,Lab --lr_file lr_file 

# python simple_train_CNN_SELEX.py RCmodel --steps train,test --tfs Scr,Ubx,Antp,Pb --lr_file lr_file 
# python simple_train_CNN_SELEX.py RCmodel --steps train,test --tfs Dfd,AbdA,AbdB,Lab --lr_file lr_file 

##### for figure 2B
# folder=/home/bxin/Documents/low_affinity/multi-module_CNN/out/SELEX_RCmodel_32filters/
# for tf in Scr Ubx Antp Pb Dfd AbdA AbdB Lab
# do
# 	cd $folder/$tf
# 	# cat simple_test_perf.txt test_perf.txt > whole_test.txt
# 	# cat simple_train_perf.txt train_perf.txt > whole_train.txt
# 	# paste -d'\t' whole_train.txt whole_test.txt | awk '{print $2"\t"$4}' - > performance_32.txt
# 	rm whole_test.txt whole_train.txt
# done

# folder=/home/bxin/Documents/low_affinity/multi-module_CNN/out/SELEX_RCmodel/
# for tf in Scr Ubx Antp Pb Dfd AbdA AbdB Lab
# do
# 	cd $folder/$tf
	#here test_perf.txt contains all parameters
	# sort -k1 test_perf.txt > whole_test.txt
	# cat simple_train_perf.txt train_perf.txt | sort -k1 - > whole_train.txt
	# paste -d'\t' whole_train.txt whole_test.txt | awk '{print $2"\t"$4}' - > performance_100.txt
# 	rm whole_train.txt whole_test.txt
# done


############### interpret models ################3
# for seq in AATGATTAATTGCT AATGATTGATTACC ATGATTTATTACCC
# do
# for method in GradientTimesInput ISM DeconvNet
# do
# 	python interpret_CNN_SELEX.py RCmodel $method --interpret_file /home/bxin/Documents/low_affinity/viz_data/$seq.h5 --tfs Ubx --lr_file ../out/SELEX_RCmodel/Ubx/train_perf.txt
# done
# done

##for figure 5A
# for tf in Scr Lab Ubx
# do
# python interpret_CNN_SELEX.py RCmodel ISM --interpret_file /home/bxin/Documents/low_affinity/viz_data/long_low_affinity_WT_short_seqs.h5 --tfs $tf --lr_file ../out/SELEX_RCmodel/$tf/best_perf.txt
# done

### for figure 5BC
# for seq in WT mut1 mut2 mut3 mut12 site1 site2 site3A site3B site3C site3D site3E
# do
# 	for tf in Scr Ubx
# 	do
# 		head -n1 ../out/SELEX_RCmodel/$tf/train_perf.txt > ../out/SELEX_RCmodel/$tf/best_perf.txt
# 		python train_CNN_SELEX.py RCmodel --steps predict --tfs $tf --pred_file /home/bxin/Documents/low_affinity/viz_data/long_low_affinity_$seq\_short_seqs.h5 --lr_file ../out/SELEX_RCmodel/$tf/best_perf.txt
# 		python train_CNN_SELEX.py RCmodel --steps deltadeltadeltaG --tfs $tf --mut_str $seq 
# done
# done

# for seq in mut1 mut2 mut3 mut12 site1 site2 site3A site3B site3C site3D site3E
# do
# 	for tf in Scr Ubx
# 	do
# 		python train_CNN_SELEX.py RCmodel --steps deltadeltadeltaG --tfs $tf --mut_str $seq 
# done
# done

### for figure 5D
#first step: rank filters of Ubx canonical models based on information content
# python train_CNN_SELEX.py RCaugmented --steps align --tfs Ubx
# matrix-clustering -v 1 -max_matrices 300 -matrix ubx filters_0,0.001,0.0003.meme meme -hclust_method average -calc sum -title 'ubx' -metric_build_tree 'Ncor' -lth w 5 -lth cor 0.6 -lth Ncor 0.4 -quick -label_in_tree name -return json,heatmap -o ./matrix-clustering 2> ./matrix-clustering_err.txt

# sort -nk7 ./matrix-clustering_tables/pairwise_compa_matrix_descriptions.tab > test # use this to figure out which one has largest IC. 

# #second step: set align_to_one_filter=True, and get aligned sequences based on one filter
# python train_CNN_SELEX.py canonical --steps align --tfs Ubx

# sort -nk2 aligned_0,0.001,0.0003_filter3.txt > aligned_0,0.001,0.0003_filter3_sorted.txt

# # replace N to n because YR code can not recognize N. 
# head -n10000 aligned_0,0.001,0.0003_filter3_sorted.txt | awk '{print $1}' - | sed s/N/n/g - > aligned_0,0.001,0.0003_filter3_low.txt
# tail -n10000 aligned_0,0.001,0.0003_filter3_sorted.txt | awk '{print $1}' - | sed s/N/n/g - > aligned_0,0.001,0.0003_filter3_high.txt
# python ../../../../codes/YRcodes.py aligned_0,0.001,0.0003_filter3_low.txt aligned_0,0.001,0.0003_filter3_low_YRcodes.txt
# python ../../../../codes/YRcodes.py aligned_0,0.001,0.0003_filter3_high.txt aligned_0,0.001,0.0003_filter3_high_YRcodes.txt

### for figure 6


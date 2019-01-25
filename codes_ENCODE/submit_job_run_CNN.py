#!/usr/bin/python
import sys
import os
import commands
import time
import csv
import re
import string

celltype = sys.argv[1]

tf_list = "/panfs/cmb-panasas2/bxin/low_affinity_CNN/" + celltype + "/tf_list_center_part2" #list of tfs, with 2 columns, or just 1 column
output_path = "/panfs/cmb-panasas2/bxin/low_affinity_CNN/" + celltype + "/"
oe_path = "/panfs/cmb-panasas2/bxin/low_affinity_CNN/out/system_io"

f = open(tf_list, "r+")
lines = f.readlines()
f.close()

# basic settings ...
mem = '15GB'
#pmem = '1000MB'
#vmem = '32GB'
wt = '120:00:00'

f = open(tf_list, "r+")
lines = f.readlines()
f.close()

for line in lines:
    newline = line.rstrip()
    items = re.split('\t', newline)
    target_tf = items[0]
    tf_len = items[1]
    tf_folder = output_path + target_tf + '/'
    # cmd = "rm /panfs/cmb-panasas2/bxin/low_affinity_CNN/out/ENCODE_canonical/" + celltype + "/" + target_tf + "/*"
    # os.system(cmd)
    qsub_filename = tf_folder + target_tf + '.run_CNN.sbatch'
    with open(qsub_filename, 'w') as qsub_file:
        qsub_file.write('#!/bin/sh\n')
        qsub_file.write('#SBATCH --ntasks=1\n')
        qsub_file.write('#SBATCH --mem=' + mem + '\n')
        qsub_file.write('#SBATCH --mem-per-cpu=' + mem + '\n')
        qsub_file.write('#SBATCH --partition=cmb\n')
        qsub_file.write('#SBATCH --constraint=sl230s\n')
        qsub_file.write('#SBATCH --time=' + wt + '\n')
        qsub_file.write('#SBATCH --error=' + oe_path + '/%x-%j.e\n')
        qsub_file.write('#SBATCH --output=' + oe_path + '/%x-%j.o\n\n')

        qsub_file.write("cd /panfs/cmb-panasas2/bxin/low_affinity_CNN/scripts\n")
	qsub_file.write("source activate py2conda\n")
       	qsub_file.write("python train_CNN_ENCODE.py canonical --steps train,test --tfs " + target_tf + " --lr_file lr_file --tf_len " + tf_len + " --celltype " + celltype + "\n")
    cmd = 'sbatch ' + qsub_filename
    os.system(cmd)

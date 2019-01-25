# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:27:25 2019

@author: bxin
"""

import sys
import os
import argparse 
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import seaborn as sns

import CNN_utils as util

def main():
    args = loadargs()
    
    for tf in args.tfs:
        ensure_dir(args.outdir)
        outdir = args.outdir +"/" + tf + "/"
        ensure_dir(outdir)
        if "train" in args.steps:
            print ("------------train:" + tf + "---------------")
            onehot_tr, hm_tr, seqs_tr, y_tr = read_data(outdir + tf + "_train.h5" )
            '''num_to_save means the number of models to be saved at the end of training'''
            '''run add_seq=True and add_hm=True models'''
            util.train_encode(outdir, onehot_tr, seqs_tr, hm_tr, y_tr, args.lr_file, args.tf_len, nfold=3, num_to_save=1)
        
        if "test" in args.steps:
            print ("------------test:"+ tf + "----------------")
            onehot_test, hm_test, seqs_test, y_test = read_data(outdir + tf + "_test.h5" )            
            util.predict_SELEX(outdir, outdir + "/train_perf.txt", onehot_test, seqs_test, hm_test, y_test)  

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
           
def loadargs():
    args = argparse.ArgumentParser(description="Generate the SELEX experiments.")
    args.add_argument("mode", type=str, help="One of canonical/RCaugmented/double.")
    args.add_argument("--steps", type=str, help="A comma-seperated list of (or subset of) train,test,predict. Default is all.")
    args.add_argument("--tf_len", type=int, help="The length of a TF's consensus sequence.")            
    args = util.parseargs("ENCODE", args)
    if args.steps == "all":
        args.steps = "train,test,predict,deltadeltadeltaG"        
    args.steps = args.steps.split(",") 
    args.outdir = args.outdir + "_" + args.mode   # ../out/ENCODE_canonical
    return args
    
def read_data(h5file):
    f = h5.File(h5file, 'r')
    g = f['data']
    data = g['s_x'].value
    seqs = g['sequence'].value
    hm = g['hm'].value
    targets = g['c0_y'].value
    return (data, hm, seqs, targets)
    
if __name__=="__main__":
    main()
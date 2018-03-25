import numpy as np
from Bio import SeqIO
import random
import re

def read_file(file):
    seqs = []
    with open(file,'r') as f:
        for seq in f:
            seqs.append(seq.strip())
    return seqs

def read_files(posfile, negfile):
    pos_seqs = []
    with open(posfile,'r') as f:
        for seq in f:
            pos_seqs.append(seq.strip())
    #This removes the sequences from positive file
    neg_seqs = []
    true_neg = []
    for record in SeqIO.parse(negfile,"fasta"):
        neg_seqs.append(str(record.seq))
    for neg_seq in neg_seqs:
        if not any(pos_seq in neg_seq for pos_seq in pos_seqs):
            true_neg.append(neg_seq)
    #Here, I am taking every 17bp of neg seq as a unit. Ideally, I'd do
    #a permutation of all possible 17bp in the neg seq but it's
    #memory intensive so, I'm trying this first.
    X = []
    k = 0
    for i in true_neg:
        n = 0
        for j in i:
            X.append([])
            X[k].extend(i[n:n+17])
            n += 17
            k += 1
    cut_neg = [s for s in X if len(s) > 16]
    return pos_seqs, cut_neg

def dna_to_vec(sequences):
    Xt = []
    i = 0
    for base in sequences:
        Xt.append([])
        for ch in base:
            if ch == 'A':
                Xt[i].extend([0,0,0,1])
            elif ch == 'T':
                Xt[i].extend([0,0,1,0])
            elif ch == 'G':
                Xt[i].extend([0,1,0,0])
            elif ch == 'C':
                Xt[i].extend([1,0,0,0])
        i += 1
    X = np.array(Xt)
    return X
        

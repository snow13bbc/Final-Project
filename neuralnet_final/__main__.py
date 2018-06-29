from .neuralnet import neuralnet
from .io import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from optparse import OptionParser
from itertools import cycle
import random

#Setting options for arguments

parser = OptionParser()
parser.add_option("-i","--iter",action = "store",type = "int",dest = "iter", default = "100")
parser.add_option("-a","--alpha",action = "store",type = "float", dest = "alpha", default = "1")
parser.add_option("-l","--lambda",action = "store",type = "float",dest = "lam", default = "0")
parser.add_option("-t","--train",action = "store_true",dest = "operationtype",default = False)
(options,args) = parser.parse_args()

#Reading in data
posfile='rap1-lieb-positives.txt'
negfile='yeast-upstream-1k-negative.fa'
pos_seqs, cut_neg = read_files(posfile, negfile)
pos_vec, neg_vec = dna_to_vec(pos_seqs),dna_to_vec(cut_neg)

#Creating datasets for cross-validation by shuffling positive datasets
avgC = int(round(pos_vec.shape[0]/5))
np.random.shuffle(pos_vec)
pos_a = pos_vec[:avgC]
pos_b = pos_vec[avgC:2*avgC]
pos_c = pos_vec[2*avgC:3*avgC]
pos_d = pos_vec[3*avgC:4*avgC]
pos_e = pos_vec[4*avgC:]
pos_list =[pos_a,pos_b,pos_c,pos_d,pos_e]
#Creating output datasets
y_a = np.ones((pos_a.shape[0],1))
y_b = np.ones((pos_b.shape[0],1))
y_c = np.ones((pos_c.shape[0],1))
y_d = np.ones((pos_d.shape[0],1))
y_e = np.ones((pos_e.shape[0],1))
y_list = [y_a,y_b,y_c,y_d,y_e]

#Repeat the same thing for negative datasets
avgK = int(round(neg_vec.shape[0]/5))
np.random.shuffle(neg_vec)
neg_a = neg_vec[:avgK]
neg_b = neg_vec[avgK:2*avgK]
neg_c = neg_vec[2*avgK:3*avgK]
neg_d = neg_vec[3*avgK:4*avgK]
neg_e = neg_vec[4*avgK:]
neg_list =[neg_a,neg_b,neg_c,neg_d,neg_e]
#Creating prediction output datasets
yhat_a = np.zeros((neg_a.shape[0],1))
yhat_b = np.zeros((neg_b.shape[0],1))
yhat_c = np.zeros((neg_c.shape[0],1))
yhat_d = np.zeros((neg_d.shape[0],1))
yhat_e = np.zeros((neg_e.shape[0],1))
yhat_list = [yhat_a,yhat_b,yhat_c,yhat_d,yhat_e]

#Initializing Neural network
#Change the layers for 8x3x8 autoencoder
nn = neuralnet(68,200,1)
#Using training data sets and outputing the score
if options.operationtype == True:
    pos = np.concatenate(pos_list)
    neg = np.concatenate(neg_list)
    y = np.concatenate(y_list)
    yhat = np.concatenate(yhat_list)
    Xtrain = np.concatenate((pos,neg))
    ytrain = np.concatenate((y,yhat))
    y_hat = nn.train_stochastic(Xtrain,ytrain,100,10,0, pos.shape[0],neg.shape[0])

    #Generating ROC curve.
    fpr,tpr, treshold = roc_curve(ytrain,y_hat, pos_label=1) # This is for regular ROC
    plt.figure()
    roc_auc = auc(fpr,tpr)

    plt.plot(fpr,tpr,color = 'black',lw = 2,label = 'ROC curve (area = {})'.format(roc_auc))
    plt.plot([0,1],[0,1],'k--',lw=2)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend(loc = "lower right")
    plt.show()

    Xtest = read_file("rap1-lieb-test.txt") # Testing dataset
    Xtest_vec = dna_to_vec(Xtest)
    yhat_test = nn.forward(Xtest_vec) # Run data through NN
    # Save data to text file

    with open("yhat_test_snow.txt", "w") as f:
        for i, seq in enumerate(Xtest):
            print (seq + "\t" + str(yhat_test[i]) + "\t")
            f.write(seq)
            f.write("\t")
            f.write(str(yhat_test[i]))
            f.write("\t")

if options.operationtype == False:
    for i in range(0,5):
    print("Training dataset {}".format(i+1))
    nn.__init__(68,200,1) # Reinitialize the neural network class so that each withheld dataset gets a fresh neural net.
    # Choose withheld dataset based on i.
    pos_test = pos_list[i]
    neg_test = neg_list[i]
    y_test = y_list[i]
    yhat_test = yhat_list[i]
    Xtrainlistpos = []
    Xtrainlistneg = []
    ytrainlistpos = []
    ytrainlistneg = []
    for j in range(0,5):
        if j != i:
            # Choose training dataset (all j for j !=i)
            Xtrainlistpos.append(pos_list[j])
            Xtrainlistneg.append(neg_list[j])
            ytrainlistpos.append(y_list[j])
            ytrainlistneg.append(yhat_list[j])

    Xtrainpos = np.concatenate(Xtrainlistpos)
    Xtrainneg = np.concatenate(Xtrainlistneg)
    ytrainpos = np.concatenate(ytrainlistpos)
    ytrainneg = np.concatenate(ytrainlistneg)
    Xtrain = np.concatenate((Xtrainpos,Xtrainneg))
    ytrain = np.concatenate((ytrainpos,ytrainneg))

    Xtest = np.concatenate((pos_test,neg_test))
    ytest = np.concatenate((y_test,yhat_test))
    # Get scores from nn.forward for test dataset, then generate and ROC curve.
    scores = nn.forward(Xtest)
    fpr,tpr,thresholds = metrics.roc_curve(ytest,scores)
    roc_auc = metrics.auc(fpr,tpr)
    lw = 2
    colors = cycle(['aqua','darkorange','cornflowerblue','darkred','black'])

    plt.plot(fpr,tpr,color = 'black',lw = lw,label = 'ROC curve (area = {})'.format(roc_auc))
    plt.plot([0,1],[0,1],'k--',lw=lw)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver operating characteristics')
    plt.legend(loc = "lower right")
    plt.show()

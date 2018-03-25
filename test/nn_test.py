from neuralnet_final import neuralnet
import numpy as np

def test_nn_autoencoder():
    pox_X = np.array([[1,1,1,1,1,1,1,1]])
    neg_X = np.array([[0,0,0,0,0,0,0,0]])

    pos_y = np.ones((pox_X.shape))
    neg_y = np.zeros((neg_X.shape))

    X = np.concatenate((pox_X,neg_X))
    y = np.concatenate((pos_y,neg_y))

    Xtest_pos = ([[1,1,1,1,1,1,1,1]])
    Xtest_neg = ([[0,0,0,0,0,0,0,0]])

    out_test_list_1 = []
    out_test_list_2 = []
    nn = neuralnet.neuralnet(4,8,1)

    for i in range(0,5):
        nn.__init__(8,3,8)
        nn.train(X,y,10000,20,0)
        out_test_1 = nn.forward(Xtest_pos)
        out_test_list_1.append(out_test_1)
        out_test_2 = nn.forward(Xtest_neg)
        out_test_list_2.append(out_test_2)

    assert np.average(out_test_list_1) > 0.9 and np.average(out_test_list_2) < 0.1

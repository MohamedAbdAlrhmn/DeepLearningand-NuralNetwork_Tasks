import math
from PreProcessing import *
import numpy as np
import sys


class backPropagation:
    def __init__(self):
        self.X = []
        self.Y = []

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    def weightInitialization(self, HiddenNodes,bias):
        weights = []
        temp1 = []
        # initialize weights for input layer
        for i in range(HiddenNodes[0]):
            temp2 = [np.random.random_sample() for i in range(5 + 1)]
            temp1.append(temp2)
        weights.append(temp1)
        # initialize weights for hidden layers
        for j in range(len(HiddenNodes) - 1):
            temp1 = []
            for i in range(HiddenNodes[j + 1]):
                temp2 = [np.random.random_sample() for i in range(HiddenNodes[j] + 1)]
                temp1.append(temp2)
            weights.append(temp1)
        temp1 = []
        # initialize weights for output layer
        for i in range(3):
            temp2 = [np.random.random_sample() for i in range(HiddenNodes[-1] + 1)]
            temp1.append(temp2)
        weights.append(temp1)
        return weights
    def pre_for_TrainAndTest(self):
        pre = PreProcessing()
        pre.preprocessing()
        Trian_Features = [pre.X_train["bill_length_mm"], pre.X_train["bill_depth_mm"], pre.X_train["flipper_length_mm"],pre.X_train["gender"], pre.X_train["body_mass_g"]]
        Test_Features = [pre.X_test["bill_length_mm"], pre.X_test["bill_depth_mm"], pre.X_test["flipper_length_mm"],pre.X_test["gender"], pre.X_test["body_mass_g"]]
        test_label = pre.Y_test
        test_label = np.array(test_label)
        train_lable = pre.Y_train
        train_lable = np.array(train_lable)
        return Trian_Features,Test_Features,train_lable,test_label
    def forwardStep(self,HiddenNodes,Test_Features,Activation_fun,test_label,w,bias,count):
        all_Net = []
        Vh = []
        all_Segma = []
        for i in range(len(HiddenNodes)):
            temp_net = [0 for i in range(HiddenNodes[i])]
            all_Net.append((temp_net))
            Vh.append([0 for i in range(HiddenNodes[i])])
            all_Segma.append([0 for i in range(HiddenNodes[i])])
        temp_net = [0 for i in range(3)]
        all_Net.append(temp_net)
        Vh.append([0 for i in range(3)])
        all_Segma.append([0 for i in range(3)])

        for i in range(len(w[0])):
            all_Net[0][i] = bias * w[0][i][0] + Test_Features[0][count] * w[0][i][1] + Test_Features[1][count] * \
                            w[0][i][2] + Test_Features[2][count] * w[0][i][3] + Test_Features[3][count] * w[0][i][4] + \
                            Test_Features[4][count] * w[0][i][5]

            if Activation_fun == "S":
                Vh[0][i] = self.sigmoid(all_Net[0][i])
            else:
                Vh[0][i] = self.Tanh(all_Net[0][i])

        for i in range(1, len(w)):
            for j in range(len(w[i])):
                temp = 0
                for k in range(len(w[i - 1]) + 1):
                    if k == 0:
                        temp += bias * w[i][j][k]
                    else:
                        temp += Vh[i - 1][k - 1] * w[i][j][k]
                all_Net[i][j] = temp
                if Activation_fun == "S":
                    Vh[i][j] = self.sigmoid(all_Net[i][j])
                else:
                    Vh[i][j] = self.Tanh(all_Net[i][j])
        for i in range(3):
            if Activation_fun == "S":
                all_Segma[-1][i] = (test_label[count][i] - Vh[-1][i]) * self.sigmoid_deriv(Vh[-1][i])
            else:
                all_Segma[-1][i] = (test_label[count][i] - Vh[-1][i]) * self.Tanh_deriv(Vh[-1][i])
        return Vh , all_Segma
    def backwardStep(self,HiddenNodes,Activation_fun,Vh,all_Segma,Test_Features,w,LR,bias,count):
        for i in reversed(range(len(HiddenNodes))):
            for j in range(HiddenNodes[i]):
                if Activation_fun == "S":
                    D_net = self.sigmoid_deriv(Vh[i][j])
                else:
                    D_net = self.Tanh_deriv(Vh[i][j])
                S = 0
                for z in range(len(w[i + 1])):
                    S += all_Segma[i + 1][z] * w[i + 1][z][j + 1]
                all_Segma[i][j] = D_net * S
        #Updating weights
        for j in range(len(w[0])):
            for k in range(len(w[0][j])):
                if k == 0:
                    w[0][j][k] = w[0][j][k] + LR * bias * all_Segma[0][j]
                else:
                    w[0][j][k] = w[0][j][k] + LR * Test_Features[k - 1][count] * all_Segma[0][j]
        for i in range(1, len(w)):
            for j in range(len(w[i])):
                for k in range(len(w[i][j])):
                    if k == 0:
                        w[i][j][k] = w[i][j][k] + LR * bias * all_Segma[i][j]
                    else:
                        w[i][j][k] = w[i][j][k] + LR * Vh[i - 1][k - 1] * all_Segma[i][j]
    def backPropagation(self, HiddenNodes, bias, Activation_fun, LR, Epochs):

        Trian_Features, Test_Features, train_lable, test_label = self.pre_for_TrainAndTest()
        w = self.weightInitialization(HiddenNodes,bias)
        # w = [[[0.1,0.5,0.4,0.9,0.3,0.88],[0.77,0.79,0.6,0.1,0.2,0.3]],[[0.8,0.95,0.4],[0.7,0.8,0.2],[0.65,0.4,0.8],[0.55,0.78,0.35]],[[0.4,0.99,0.7,0.6,0.42],[0.76,0.1,0.9,0.7,0.47],[0.4,0.3,0.9,0.2,0.63]]]
        count = 0
        c = 0
        Tcf_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for E in range(Epochs):
            Vh, all_Segma = self.forwardStep(HiddenNodes,Test_Features,Activation_fun,test_label,w,bias,count)
            c = self.Accuracy(Vh,test_label,c,count)

            confusion_matrix,TP1,TP2,TP3,TN1,TN2,TN3,FP1,FP2,FP3,FN1,FN2,FN3 = self.confusion_matrix(np.array(test_label[count]).tolist(), Vh[-1], Tcf_matrix,0,0,0,0,0,0,0,0,0,0,0,0)

            self.backwardStep(HiddenNodes,Activation_fun,Vh,all_Segma, Test_Features, w, LR, bias, count)
            count += 1
            count = count % len(Test_Features[0])
        print("Training confusion_matrix = ", confusion_matrix)
        print("TP1 = ", TP1, "TP2 = ", TP2, "TP3 = ", TP3)
        print("TN1 = ", TN1, "TN2 = ", TN2, "TN3 = ", TN3)
        print("FP1 = ", FP1, "FP2 = ", FP2, "FP3 = ", FP3)
        print("FN1 = ", FN1, "FN2 = ", FN2, "FN3 = ", TN3)
        print("training Accurecy = ", c / Epochs * 100)

        return w,test_label,Test_Features
    def Accuracy(self,Vh,test_label,true,e):
        max_VH = max(Vh[-1])
        for i in range(len(Vh[-1])):
            if Vh[-1][i] == max_VH:
                Vh[-1][i] = 1
            else:
                Vh[-1][i] = 0
        if Vh[-1] == np.array(test_label[e]).tolist():
            true += 1
        return true
    def Test(self, HiddenNodes, bias, Activation_fun, LR, Epochs):
        w,test_label,Test_Features = self.backPropagation(HiddenNodes, bias, Activation_fun, LR, Epochs)
        true = 0
        cf_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for e in range(len(test_label)):
            Vh,all_Segma = self.forwardStep(HiddenNodes,Test_Features,Activation_fun,test_label,w,bias,e)
            true = self.Accuracy(Vh,test_label,true,e)

            confusion_matrix,TP1,TP2,TP3,TN1,TN2,TN3,FP1,FP2,FP3,FN1,FN2,FN3 = self.confusion_matrix(np.array(test_label[e]).tolist(),Vh[-1],cf_matrix,0,0,0,0,0,0,0,0,0,0,0,0)
        print("Testing confusion_matrix = ",confusion_matrix )

        print("TP1 = ", TP1, "TP2 = ", TP2, "TP3 = ", TP3)
        print("TN1 = ", TN1, "TN2 = ", TN2, "TN3 = ", TN3)
        print("FP1 = ", FP1, "FP2 = ", FP2, "FP3 = ", FP3)
        print("FN1 = ", FN1, "FN2 = ", FN2, "FN3 = ", FN3)
        print("Testing Accurecy = ", true / len(test_label) * 100)



    def confusion_matrix(self,target,y,cf_matrix,TP1,TP2,TP3,TN1,TN2,TN3,FP1,FP2,FP3,FN1,FN2,FN3):

        if target == [1,0,0] and y ==[1,0,0]:
            cf_matrix[0][0] += 1
        elif target == [1,0,0] and y == [0,1,0]:
            cf_matrix[0][1] += 1
        elif target == [1,0,0] and y == [0,0,1]:
            cf_matrix[0][2] += 1
        elif target == [0,1,0] and y == [1,0,0]:
            cf_matrix[1][0] += 1
        elif target == [0,1,0] and y == [0,1,0]:
            cf_matrix[1][1] += 1
        elif target == [0,1,0] and y == [0,0,1]:
            cf_matrix[1][2] += 1
        elif target == [0,0,1] and y == [1,0,0]:
            cf_matrix[2][0] += 1
        elif target ==[0,0,1] and y == [0,1,0]:
            cf_matrix[2][1] += 1
        elif target == [0,0,1] and y == [0,0,1]:
            cf_matrix[2][2] += 1
        #TP1,TP2,TP3,TN1,TN2,TN3,FP1,FP2,FP3,FN1,FN2,FN3
        TP1 += cf_matrix[0][0]
        FP1 += cf_matrix[1][0] + cf_matrix[2][0]
        TN1 += cf_matrix[1][1] + cf_matrix[1][2] + cf_matrix[2][1] + cf_matrix[2][2]
        FN1 += cf_matrix[0][1] + cf_matrix[0][2]

        TP2 += cf_matrix[1][1]
        FP2 += cf_matrix[0][1] + cf_matrix[2][1]
        TN2 += cf_matrix[0][0] + cf_matrix[0][2] + cf_matrix[2][0] + cf_matrix[2][2]
        FN2 += cf_matrix[1][0] + cf_matrix[1][2]

        TP3 += cf_matrix[2][2]
        FP3 += cf_matrix[0][2] + cf_matrix[1][2]
        TN3 += cf_matrix[0][0] + cf_matrix[0][1] + cf_matrix[1][0] + cf_matrix[1][1]
        FN3 += cf_matrix[2][0] + cf_matrix[2][1]

        return cf_matrix,TP1,TP2,TP3,TN1,TN2,TN3,FP1,FP2,FP3,FN1,FN2,FN3
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def Tanh(self, x):
        return (1 - np.exp(-x)) / (1 + np.exp(-x))

    def Tanh_deriv(self, x):
        tanh = self.Tanh(x)
        return (1 - (math.pow(tanh, 2)))
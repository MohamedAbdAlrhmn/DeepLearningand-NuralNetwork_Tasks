import numpy as np
from PreData import *
import matplotlib.pyplot as plt
from sklearn import metrics

predicted = [None] * 40
classes = [None] * 40


def Calc_NetValue_Train(bias, features, weights):
    net_value = bias * weights[0] + weights[1] * features[0] + weights[2] * features[1]
    return net_value


def Calculate_NetValue_Test(bias, features, weights):
    net_value = bias + weights[0] * features[0] + weights[1] * features[1]
    return net_value


def Signum(value):
    desired = 0
    if value >= 0:
        desired = 1
    else:
        desired = -1
    return desired


class Perceptron:

    def Train_Data(self, feature1, feature2, classes, epoch, learning_rate, bias):
        weight = [np.random.random_sample(), np.random.random_sample(), np.random.random_sample()]
        for z in range(epoch):
            for i in range(60):
                features = [feature1[i], feature2[i]]
                desired_output = classes[i]
                net_val = Calc_NetValue_Train(bias, features, weight)
                actual = Signum(net_val)
                error = desired_output - actual
                weight[0] += learning_rate * error
                weight[1] += learning_rate * error * features[0]
                weight[2] += learning_rate * error * features[1]

        last_weight1 = weight[1]
        last_weight2 = weight[2]
        last_bias = weight[0] * bias
        return last_weight1, last_weight2, last_bias

    def Test_Data(self, f1, f2, classes, weight1, weight2, bias):

        weight = [weight1, weight2]
        bias = bias
        correct = 0
        wrong = 0

        for i in range(40):
            features = [f1[i], f2[i]]
            net_input = Calculate_NetValue_Test(bias, features, weight)
            actual_output = Signum(net_input)
            predicted[i] = actual_output
            if actual_output == classes[i]:
                correct += 1
            else:
                wrong += 1

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(40):
            if predicted[i] == classes[i]:
                if classes[i] == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if classes == -1:
                    fp += 1
                else:
                    fn += 1
        confusionmatrix = [
            [tn, fp],
            [fn, tp]
        ]
        confusionmatrix = np.array(confusionmatrix)
        print("Confusion Matrix :- ")
        print(confusionmatrix)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusionmatrix, display_labels=[False, True])

        cm_display.plot()
        plt.show()
        accuracy = (correct / 40) * 100
        return accuracy


# confusion matrix


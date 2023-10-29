from ReadFromCSV import PreProcessing


class run:

    def __int__(self, feature1, feature2, class1, class2, epoch, learning_rate, bias):
        self.feature1 = feature1
        self.feature2 = feature2
        self.class1 = class1
        self.class2 = class2
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.bias = bias

    def pre_process(self):
        read = PreProcessing()
        read.ReadCSV()
        fx = self.getFeature(self.feature1, read)
        fy = self.getFeature(self.feature2, read)
        classes = []
        for i in range(30):
            classes.append(self.class1)
        for i in range(30):
            classes.append(self.class2)

        return fx, fy, classes

    def getFeature(self, index, read):
        feature = []
        if index == 1:
            feature = read.x1
        elif index == 2:
            feature = read.x2
        elif index == 3:
            feature = read.x3
        elif index == 4:
            feature = read.x4
        else:
            feature = read.x5
        return feature

    def Pre_for_train(self):
        fx, fy, class_label = self.pre_process()
        class_train = []
        first_class = class_label[0]
        for i in class_label:
            if i == first_class:
                class_train.append(1)
            else:
                class_train.append(-1)

        fx_train = []
        fy_train = []
        if self.class1 == 1:
            fx_train = fx[0:30]
            fy_train = fy[0:30]
            if self.class2 == 2:
                fx_train.extend(fx[50:80])
                fy_train.extend(fy[50:80])
            else:
                fx_train.extend(fx[100:130])
                fy_train.extend(fy[100:130])

        elif self.class1 == 2:
            fx_train = fx[50:80]
            fy_train = fy[50:80]
            fx_train.extend(fx[100:130])
            fy_train.extend(fy[100:130])

        return fx_train, fy_train, class_train

    def pre_for_test(self):

        fx, fy, class_label = self.pre_process()
        class_test = []
        first_class = class_label[0]

        for i in range(20):
            if class_label[i] == first_class:
                class_test.append(1)
            else:
                class_test.append(-1)
        for i in range(30, 50):
            if class_label[i] == first_class:
                class_test.append(1)
            else:
                class_test.append(-1)

        fx_test = []
        fy_test = []
        if self.class1 == 1:
            fx_test = fx[30:50]
            fy_test = fy[30:50]
            if self.class2 == 2:
                fx_test.extend(fx[80:100])
                fy_test.extend(fy[80:100])
            else:
                fx_test.extend(fx[130:150])
                fy_test.extend(fy[130:150])

        elif self.class1 == 2:
            fx_test = fx[80:100]
            fy_test = fy[80:100]
            fx_test.extend(fx[130:150])
            fy_test.extend(fy[130:150])

        return fx_test, fy_test, class_test



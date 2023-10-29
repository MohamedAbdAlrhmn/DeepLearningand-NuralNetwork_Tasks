import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class PreProcessing:
    def _init_(self):
        self.X_train = []
        self.X_test = []
        self.Y_train = []
        self.Y_test = []

    def preprocessing(self):
        #read Data
        data = pd.read_csv('penguins.csv')
        data = data.reset_index(drop=True)

        # fill Null Values
        data.fillna(method="ffill", inplace=True)

        # Feature Encoding
        lbl = LabelEncoder()
        lbl.fit(list(data['gender'].values))
        data["gender"] = lbl.transform(list(data['gender'].values))
        X = data.iloc[:, 1:]
        Y = pd.get_dummies(data.species, prefix='')

        #Split Data
        x1 = X[0:50]
        x2 = X[50:100]
        x3 = X[100:150]

        y1 = Y[0:50]
        y2 = Y[50:100]
        y3 = Y[100:150]

        X1_train, X1_test, Y1_train, Y1_test = train_test_split(x1, y1, test_size=0.40, train_size=0.60, shuffle=True)
        X2_train, X2_test, Y2_train, Y2_test = train_test_split(x2, y2, test_size=0.40, train_size=0.60, shuffle=True)
        X3_train, X3_test, Y3_train, Y3_test = train_test_split(x3, y3, test_size=0.40, train_size=0.60, shuffle=True)

        #Concatinate Splited Data
        self.X_train = pd.concat([X1_train, X2_train, X3_train], ignore_index=True)
        self.X_test = pd.concat([X1_test, X2_test, X3_test], ignore_index=True)
        self.Y_train = pd.concat([Y1_train, Y2_train, Y3_train], ignore_index=True)
        self.Y_test = pd.concat([Y1_test, Y2_test, Y3_test], ignore_index=True)


        #Shuffle All Data
        tmp = []
        tmp1 = []
        features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "gender", "body_mass_g"]
        classes = ["_Adelie","_Gentoo","_Chinstrap"]
        for i in range(90):
            tmp.append(i)
        random.shuffle(tmp)

        for i in range(90):
            for j in range(5):
                self.X_train[features[j]][i] = self.X_train[features[j]][tmp[i]]

        for i in range(90):
            for j in range(3):
                self.Y_train[classes[j]][i] = self.Y_train[classes[j]][tmp[i]]
        for i in range(60):
            tmp1.append(i)
        random.shuffle(tmp1)

        for i in range(60):
            for j in range(5):
                self.X_test[features[j]][i] = self.X_test[features[j]][tmp1[i]]
        for i in range(60):
            for j in range(3):
                self.Y_test[classes[j]][i] = self.Y_test[classes[j]][tmp1[i]]

        # Normalization
        for i in range(len(self.X_train)):
            for j in features:
                self.X_train[j][i] = (self.X_train[j][i] - min(self.X_train[j])) / (
                            max(self.X_train[j]) - min(self.X_train[j]))
        for i in range(len(self.X_test)):
            for j in features:
                self.X_test[j][i] = (self.X_test[j][i] - min(self.X_test[j])) / (
                            max(self.X_test[j]) - min(self.X_test[j]))

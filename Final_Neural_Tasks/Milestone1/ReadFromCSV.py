import matplotlib.pyplot as plt


class PreProcessing:
    def __init__(self):
        self.x1 = []
        self.x2 = []
        self.x3 = []
        self.x4 = []
        self.x5 = []
        self.label = []

    def ReadCSV(self):
        ok = 0
        with open('penguins.csv') as Row:
            Row.readline()
            for r in Row:
                column = r.split(',')
                self.x1.append(float(column[1]))
                self.x2.append(float(column[2]))
                self.x3.append(float(column[3]))
                if column[4] == "NA" and ok == 1:
                    column[4] = "male"
                    self.x4.append(1)
                    ok = 0
                elif column[4] == "NA" and ok == 0:
                    column[4] = "female"
                    self.x4.append(0)
                    ok = 1
                else:
                    if column[4] == "male":
                        self.x4.append(1)
                    else:
                        self.x4.append(0)
                self.x5.append(float(column[5]))
                if column[0] == "Adelie":
                    self.label.append(1)
                elif column[0] == "Gentoo":
                    self.label.append(2)
                else:
                    self.label.append(3)

        for i in range(150):
            self.x1[i] = (self.x1[i] - min(self.x1)) / (max(self.x1) - min(self.x1))
            self.x2[i] = (self.x2[i] - min(self.x2)) / (max(self.x2) - min(self.x2))
            self.x3[i] = (self.x3[i] - min(self.x3)) / (max(self.x3) - min(self.x3))
            self.x4[i] = (self.x4[i] - min(self.x4)) / (max(self.x4) - min(self.x4))
            self.x5[i] = (self.x5[i] - min(self.x5)) / (max(self.x5) - min(self.x5))

    def visualization(self):
        plt.figure('fig1')
        plt.scatter(self.x1[0:50], self.x2[0:50])
        plt.scatter(self.x1[50:100], self.x2[50:100])
        plt.scatter(self.x1[100:150], self.x2[100:150])

        plt.figure('fig2')
        plt.scatter(self.x1[0:50], self.x3[0:50])
        plt.scatter(self.x1[50:100], self.x3[50:100])
        plt.scatter(self.x1[100:150], self.x3[100:150])

        plt.figure('fig3')
        plt.scatter(self.x1[0:50], self.x4[0:50])
        plt.scatter(self.x1[50:100], self.x4[50:100])
        plt.scatter(self.x1[100:150], self.x4[100:150])

        plt.figure('fig4')
        plt.scatter(self.x1[0:50], self.x5[0:50])
        plt.scatter(self.x1[50:100], self.x5[50:100])
        plt.scatter(self.x1[100:150], self.x5[100:150])

        plt.figure('fig5')
        plt.scatter(self.x2[0:50], self.x3[0:50])
        plt.scatter(self.x2[50:100], self.x3[50:100])
        plt.scatter(self.x2[100:150], self.x3[100:150])

        plt.figure('fig6')
        plt.scatter(self.x2[0:50], self.x4[0:50])
        plt.scatter(self.x2[50:100], self.x4[50:100])
        plt.scatter(self.x2[100:150], self.x4[100:150])

        plt.figure('fig7')
        plt.scatter(self.x2[0:50], self.x5[0:50])
        plt.scatter(self.x2[50:100], self.x5[50:100])
        plt.scatter(self.x2[100:150], self.x5[100:150])

        plt.figure('fig8')
        plt.scatter(self.x3[0:50], self.x4[0:50])
        plt.scatter(self.x3[50:100], self.x4[50:100])
        plt.scatter(self.x3[100:150], self.x4[100:150])

        plt.figure('fig9')
        plt.scatter(self.x3[0:50], self.x5[0:50])
        plt.scatter(self.x3[50:100], self.x5[50:100])
        plt.scatter(self.x3[100:150], self.x5[100:150])

        plt.figure('fig10')
        plt.scatter(self.x4[0:50], self.x5[0:50])
        plt.scatter(self.x4[50:100], self.x5[50:100])
        plt.scatter(self.x4[100:150], self.x5[100:150])

        plt.show()

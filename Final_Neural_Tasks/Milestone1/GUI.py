from tkinter import *
from tkinter import messagebox
import matplotlib.pyplot as plt
from ReadFromCSV import *
from PreData import run
from Perceptron import *

check = dict()
check["bill_length"] = 1
check["bill_depth"] = 2
check["flipper_length"] = 3
check["gender"] = 4
check["body_mass"] = 5
check["Adelie"] = 1
check["Gentoo"] = 2
check["Chinstrap"] = 3

window = Tk()
tst = run()
algo = Perceptron()
window.title("Task1_deeplearning")
window.minsize(400, 400)
window.configure(bg='grey')
# featur1_lablel=Label(text="Feature1")
# featur1_lablel.pack()
# featur1_entry=Entry()
# featur1_entry.pack()
# def display_content():
#     content=featur1_entry.get()
#     result_lable = Label(text="result is" + str(content))
#     result_lable.pack()
# btn=Button(text="Run",command=display_content)
# btn.pack()

featur1_lablel = Label(text="Feature1")
featur1_lablel.pack()
# feature1 is string and save value that will choosen
featur1_menue = StringVar()
featur2_menue = StringVar()
class1_menue = StringVar()
class2_menue = StringVar()
# ==================================================================
Options = ["bill_length", "bill_depth", "flipper_length", "gender", "body_mass"]
class_List = ["Adelie", "Gentoo", "Chinstrap"]
Feature1 = OptionMenu(window, featur1_menue, *Options)
Feature1.pack()

featur2_lablel = Label(text="Feature2")
featur2_lablel.pack()

Feature2 = OptionMenu(window, featur2_menue, *Options)
Feature2.pack()

# ==========================================================================

class1_label = Label(text="First_Class")
class1_label.pack()

Entry_Class1 = OptionMenu(window, class1_menue, *class_List)
Entry_Class1.pack()

class2_label = Label(text="Second_Class")
class2_label.pack()

Entry_Class2 = OptionMenu(window, class2_menue, *class_List)
Entry_Class2.pack()

# Test_class2_entry
# def display1():
#     class2_menue.get()
#     label1=Label(window,text=class2_menue.get()).pack()
#
# btn=Button(window,text="content",command=display1).pack()
# =============================================================
iterations = Label(text="Number_OF_Epocs")
iterations.pack()

Entry_Epoc = Entry()
Entry_Epoc.pack()

# test_epoc
# def display():
#     label1 = Label(window, text=Entry_Epoc.get()).pack()
# btn=Button(window,text="content",command=display).pack()
# =========================================================
LRate = Label(text="Learning_Rate")
LRate.pack()

LearningRate_entry = Entry()
LearningRate_entry.pack()

# test_Learningrate
# def display():
#     label1 = Label(window, text=LearningRate_entry.get()).pack()
# btn=Button(window,text="content",command=display).pack()

# ================================================================================================
# test_radio button
# def on_click(value):
#     my_lable=Label(window,text=value).pack()

value_of_button = IntVar()
# Radiobutton(window,text="Bias",variable=value_of_button,value=1,command=lambda :on_click(value_of_button.get())).pack()
# Radiobutton(window,text="No_Bias",variable=value_of_button,value=0,command=lambda :on_click(value_of_button.get())).pack()

Radiobutton(window, text="Bias", variable=value_of_button, value=1).pack()
Radiobutton(window, text="No_Bias", variable=value_of_button, value=0).pack()


def click_button():
    class1 = class1_menue.get()
    class2 = class2_menue.get()
    F1 = featur1_menue.get()
    F2 = featur2_menue.get()
    epoc = Entry_Epoc.get()
    learning_rate = LearningRate_entry.get()
    radio_buton = value_of_button.get()

    try:
        epoc = int(epoc)
        learning_rate = float(learning_rate)
        tst.feature1 = check[F1]
        tst.feature2 = check[F2]
        tst.class1 = check[class1]
        tst.class2 = check[class2]
        tst.epoch = epoc
        tst.learning_rate = learning_rate
        tst.bias = radio_buton

        x1, x2, clacc = tst.Pre_for_train()
        indicis = []
        for i in range(0, 60):
            indicis.append(i)
        np.random.shuffle(indicis)

        for i in range(60):
            x1[i] = x1[indicis[i]]
            x2[i] = x2[indicis[i]]
            clacc[i] = clacc[indicis[i]]
        w1, w2, bias = algo.Train_Data(x1, x2, clacc, tst.epoch, tst.learning_rate, tst.bias)
        f1, f2, classes = tst.pre_for_test()
        new_ind = []
        for i in range(0, 40):
            new_ind.append(i)

        for i in range(40):
            f1[i] = f1[new_ind[i]]
            f2[i] = f2[new_ind[i]]
            classes[i] = classes[new_ind[i]]

        l1=[]
        l2=[]
        l3=[]
        l4=[]

        for i in range(len(f1)):
            if (classes[i] == 1):
                l1.append((f1[i]))
                l2.append((f2[i]))
            else:
                l3.append((f1[i]))
                l4.append((f2[i]))

        x_test=f1+f2

        x=[min(x_test),max(x_test)]
        y = [ -(bias  + w1 * xi) / w2 for xi in x]
        plt.scatter(l1,l2)
        plt.scatter(l3,l4)
        plt.plot(x,y)
        plt.show()



        acc = algo.Test_Data(f1, f2, classes, w1, w2, bias)

        print("Accuracy :", acc,"%")
        # obj = PreProcessing()
        # obj.ReadCSV()
        # obj.visualization()



    except ValueError:
        messagebox.showinfo("Error", "Please, Enter the valid number")

def visual_graph():
    obj = PreProcessing()
    obj.ReadCSV()
    obj.visualization()




start_button = Button(window, text="Visualization", background="white", command=visual_graph).pack()
Visualization_before=Button(window,text="Test_Graph",background="white",command=click_button).pack()


# ====================================================================================


window.mainloop()






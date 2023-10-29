from backPropagation import backPropagation
Activation_fun = 'S'
# LR = 0.1
# Epochs = 50000
# NumLayers = 1
# HiddenNodes = [3]
# bias = "y"
#
# b = backPropagation()
# w = b.Test(HiddenNodes, 1, Activation_fun, LR, Epochs)
from tkinter import messagebox
from tkinter import *
from tkinter import ttk

from backPropagation import backPropagation

def create_cmb(row, col, values):
    feature_cmb = ttk.Combobox(top, values=values, width=50)
    feature_cmb['state'] = 'readonly'
    feature_cmb.grid(column=col, row=row)
    return feature_cmb


def create_label(text, row, col):
    label = Label(top, text=text, font=40)
    label.grid(column=col, row=row, sticky=W, pady=10)


# Graphical User Interface
top = Tk()
top.title("penguins detection")
top.geometry("750x600")

Options = ["Sigmoid", "Tanh"]

# _______________________________________________
create_label("_________________", 1, 0)

create_label("Number of hidden layers", 3, 0)
e1 = Entry(top, bd=5)
e1.grid(row=3, column=4)

create_label("Number of neurons", 5, 0)
e2 = Entry(top, bd=5)
e2.grid(row=5, column=4)

create_label("Learning rate", 7, 0)
e3 = Entry(top, bd=5)
e3.grid(row=7, column=4)

create_label("Epochs", 9, 0)
e4 = Entry(top, bd=5)
e4.grid(row=9, column=4)
# checkvar = IntVar()
# chkbtn = Checkbutton(top, text="Bias", variable=checkvar).place(x=300, y=320)
check_var = IntVar()
bias_check = Checkbutton(top, variable=check_var, text="Add bias",
                         onvalue=1, offvalue=0, height=4, width=10, font=30)
bias_check.grid(row=15, column=1)

create_label("Activation function", 11, 0)

ActivationFunction = create_cmb(12, 0, Options)


def click_button():
    hls = e1.get()
    nns = e2.get()
    etta = e3.get()
    eps = e4.get()
    bis = check_var.get()
    acf = ActivationFunction.get()
    neurons_list = nns.split(",")
    neurons_list = list(map(int,neurons_list))

    try:
        hls = int(hls)
        etta = float(etta)
        Activation_fun = acf[0].upper()
        LR = etta
        Epochs = int(eps)
        NumLayers = int(hls)
        HiddenNodes = neurons_list
        bias = int(bis)
        b = backPropagation()
        b.Test(HiddenNodes, bias, Activation_fun, LR, Epochs)
        # w = b.Test(HiddenNodes, 1, Activation_fun, LR, Epochs)
    except ValueError:
        messagebox.showinfo("Error", "Please, Enter the valid number")


# b1 = Button(top, text="Visualization").place(x=345, y=400)
b1 = Button(top, text=" Model Test ", command=click_button).place(x=345, y=400)

top.mainloop()
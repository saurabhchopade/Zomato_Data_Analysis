from tkinter import *

from zomatoml import dtada_fun
from zomatoml import rf_fun
from zomatoml import sgd_fun
from zomatoml import avgall_fun
from zomatoml import compall_fun


window=Tk()
window.geometry('600x600')

l1=Label(window,text=" **Classification algorithm And Comparison**")
l1.pack()
b1=Button(window,text="ADABOOST",padx=20)
b1.config(command=dtada_fun)
b1.pack()

b2=Button(window,text="RANDOM FOREST",padx=20)
b2.config(command=rf_fun)
b2.pack()

b3=Button(window,text="SGD",padx=20)
b3.config(command=sgd_fun)
b3.pack()

b4=Button(window,text="AVERAGE OF ALL ALGO",padx=20)
b4.config(command=avgall_fun)
b4.pack()

b5=Button(window,text="Comapare Precision Recall F1score",padx=20)
b5.config(command=compall_fun)
b5.pack()

window.mainloop()

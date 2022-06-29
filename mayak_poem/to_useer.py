from tkinter import *  
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def clicked():  
    model.eval()
    lbl.configure(text=evaluate(
    model, 
    char_to_idx, 
    idx_to_char, 
    temp=0.9, 
    prediction_len=900, 
    start_text=' '
    )
)
window = Tk()  
window.title("MayakPoem")  
window.geometry('400x800')  
lbl = Label(window, text="тут будет ваш текст", font=("Arial", 10))  
lbl.grid(column=0, row=0)
btn = Button(window, text="Сгенерировать", command=clicked)  
btn.grid(column=1, row=0)  
window.mainloop()

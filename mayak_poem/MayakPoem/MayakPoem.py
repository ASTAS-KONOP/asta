from tkinter import *  

def clicked():  
    lbl.configure(text='     просты\n                       на облаком,\n                          и дих окр.\n           Нос призселое трионтаригашеть,\n             на спусканий.\n            Молке были известной надо\n                                свои сизовенности».\n          А вы ж мекло\n                             тим.\n                                     А городу\n                                      помолстынучки!\n              Непорохом мрежды,\n           и отсталось комтого,\n                                 мы был вы!\n    Мы\n           альбодноа марклопить\n                                   воде нам\n                        ваших сердце хотил:\n     О!!\n      «Вечернока сторожья.\n                                   Ленин да семья.\n    Владимирования в вор,\n                                    попросовала\n       под мерые\n                  и вдети,\n          черт\n Утроник киломсоломощные\n')    

window = Tk()  
window.title("MayakPoem")  
window.geometry('400x800')  
lbl = Label(window, text="тут будет ваш текст", font=("Arial", 10))  
lbl.grid(column=0, row=0)
btn = Button(window, text="Сгенерировать", command=clicked)  
btn.grid(column=1, row=0)  
window.mainloop()

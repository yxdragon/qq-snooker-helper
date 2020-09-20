import tkinter as tk
import numpy as np
import extract, robot, table
from threading import Thread
from time import time, sleep

config = {'mode':'black8', 'iter':1, 'goal':-1, 'run':False}

def show_table():
    img = np.array(robot.grab())[:,:,:3]
    tab = extract.extract_table(img, config['mode'])
    if isinstance(tab, str):
        print(tab)
    else:
        tab = table.Table(*tab, config['mode'])
        tab.solve(-1, 2)
        tab.show()

def set_info(cont):
    lab.config(text=cont)

def hold(tp='black8', goal=-1, maxiter=1, call=set_info):
    while True:
        sleep(0.5)
        if not config['run']:
            set_info('QQ桌球瞄准器')
            continue
        img = np.array(robot.grab())[:,:,:3]
        table, note = robot.analysis(img,
            config['mode'], config['goal'], config['iter']+1)
        if table is None:
            call(note)
            continue
        else:
            call(note)
            robot.snap(table, call)

def on_stop(): config['run'] = False

def on_start(): config['run'] = True

def on_iter():
    config['iter'] = (config['iter'] + 1)%3
    btn_iter.config(text='传击:%d次'%(config['iter']+1))

def on_mode():
    if config['mode']=='black8':
        config['mode'] = 'snooker'
        btn_mode.config(text='模式: 斯 诺 克')
    elif config['mode']=='snooker':
        config['mode'] = 'black8'
        btn_mode.config(text='模式:中式黑八')
    
if __name__ == '__main__':
    top = tk.Tk()
    top.attributes("-topmost", True)
    top.title('QQ桌球瞄准器')

    def helloCallBack():
       print( "Hello Python", "Hello Runoob")
     
    btn_start = tk.Button(top, text ="开始", command = on_start)
    btn_stop = tk.Button(top, text ="暂停", command = on_stop)

    btn_iter = tk.Button(top, text ="传击:2次", command = on_iter)
    btn_mode = tk.Button(top, text ="模式:中式黑八", command = on_mode)


    var = tk.StringVar()
    rad_snooker = tk.Radiobutton(top, text='斯诺克', variable=var, value='A', command=helloCallBack)
    rad_black8 = tk.Radiobutton(top, text='中八', variable=var, value='B', command=helloCallBack)
    lab = tk.Label(top, text='桌球瞄准器', bg='white')
    btn_plot = tk.Button(top, text ="绘制", command = show_table)

    lab.pack(side = 'bottom', fill='x')
    btn_start.pack(side='left')
    btn_stop.pack(side='left')
    btn_plot.pack(side='left')
    btn_iter.pack(side='left')
    #rad_snooker.pack(side='left')
    #rad_black8.pack(side='left')
    btn_mode.pack(side='left')
    thread = Thread(target=hold)
    thread.setDaemon(True)
    thread.start()
    top.mainloop()

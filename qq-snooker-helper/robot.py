import win32api, win32con
from time import sleep
from PIL import ImageGrab, Image
import numpy as np
from time import time, sleep
from numpy.linalg import norm
from table import Table

def move(x, y):
    win32api.SetCursorPos([int(round(x)), int(round(y))])
    
def right(x, y):
    win32api.SetCursorPos([x, y])
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP | win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)

def left(x, y):
    win32api.SetCursorPos([x, y])
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP | win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

def getpos():
    return win32api.GetCursorPos()

def grab():
    return ImageGrab.grab()

def snap(table, call=print):
    if len(table.hitpts)==0: return
    x, y = getpos()
    pts = table.hitpts[:,:2] + table.loc
    dif = norm(pts-(y, x), axis=-1)
    if dif.min()<8:
        n = np.argmin(dif)
        y, x = pts[n]
        obj = table.hitpts[n]
        move(x, y)
        call('锁定目标，传递%d次，成功率%d%%'%(obj[3],obj[4]))
    
def analysis(img, tp='black8', goal=-1, maxiter=2):
    #img = Image.open('testimg/black8.png')
    from extract import extract_table
    table = extract_table(img, tp)
    if isinstance(table, str): 
        return (None, table)
    table = Table(*table, tp)
    table.solve(goal, maxiter)
    return (table, '共检测到%s条击球策略'%len(table.hitpts))

def hold(tp='black8', goal=-1, maxiter=2, call=print):
    while True:
        sleep(0.5)
        img = np.array(grab())[:,:,:3]
        table, note = analysis(img, tp, goal, maxiter)
        if table is None:
            call(note)
            continue
        else:
            call(note, call)
            snap(table)
                
if __name__ == '__main__':
    hold()

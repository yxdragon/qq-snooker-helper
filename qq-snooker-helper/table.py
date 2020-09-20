import matplotlib.pyplot as plt
from skimage.io import imread
from time import time
import numpy as np
from math import sin, cos, pi, asin, acos
pi2, pi = pi * 2, pi

def anglex(dx, dy):
    a = acos(dx / (dx**2 + dy**2)**0.5)
    return a if dy > 0 else pi * 2 - a

class Arc:
    def __init__(self, a1, a2):
        self.a1, self.a2 = a1%pi2, a2%pi2

    def value(self):
        dv = (self.a2 - self.a1) % pi2
        return dv if dv < pi else dv - pi2

    def mid(self):
        v = self.a1 + self.value()/2
        return v % (np.pi*2)

    def contain(self, v):
        da1 = (v - self.a1) % pi2
        da2 = (self.a2 - v) % pi2
        d12 = (self.a2-self.a1)%pi2
        return abs(da1+da2-d12)<1e-5
    
    def minus(self, arc):
        if arc.contain(self.a1):
            self.a1 = arc.a2
        if arc.contain(self.a2):
            self.a2 = arc.a1
        return self.value() > 0

    def __str__(self):
        return 'Arc(%.2f,%.2f)'%(self.a1, self.a2)

class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2

    def __str__(self):
        return '%s %s'%(self.p1, self)

class Path:
    def __init__(self, frm, to, arc, l, layer=0):
        self.frm, self.to, self.arc, self.l = frm, to, arc, l
        self.line, self.point, self.layer = None, None, layer

    def shadow(self, ball):
        dv_x, dv_y = ball.x - self.frm.x, ball.y - self.frm.y
        a = anglex(dv_x, dv_y)
        l = (dv_x**2 + dv_y**2)**0.5
        if l > self.l : return False
        da = asin(min(self.frm.r*2/l, 1))
        arc = Arc(a-da, a+da)
        self.arc.minus(arc)

    def inpoint(self, ball):
        x, y = self.get_point()
        if ((ball.x-x)**2+(ball.y-y)**2)**0.5<ball.r*2:
            self.arc.a1, self.arc.a2 = 0.1, -0.1

    def get_line(self):
        if not self.line is None: return self.line
        x, y, arc, r = self.frm.x, self.frm.y, self.arc, self.frm.r*2
        x1, y1 = cos(arc.a1+pi) * r + x, sin(arc.a1+pi) * r + y
        x2, y2 = cos(arc.a2+pi) * r + x, sin(arc.a2+pi) * r + y
        self.line = Line(x2, y2, x1, y1)
        return self.line

    def get_point(self):
        if not self.point is None: return self.point
        x, y, arc, r = self.frm.x, self.frm.y, self.arc, self.frm.r
        x, y = cos(arc.mid()) * self.l + x, sin(arc.mid()) * self.l + y
        self.point = (x, y)
        return self.point

class Ball:
    def __init__(self, x, y, r, tp):
        self.x, self.y, self.r, self.tp = x, y, r, tp

    def distance(self, ball):
        return ((ball.x - self.x)**2 + (ball.y - self.y)**2)**0.5

    def to_line(self, line, buf=False):
        isline = isinstance(line, Line)
        ln = line if isline else line.get_line()
        dv1_x, dv1_y = ln.x1 - self.x, ln.y1 - self.y
        dv2_x, dv2_y = ln.x2 - self.x, ln.y2 - self.y
        a1 = anglex(dv1_x, dv1_y)
        a2 = anglex(dv2_x, dv2_y)
        l1 = (dv1_x**2 + dv1_y**2)**0.5
        l2 = (dv2_x**2 + dv2_y**2)**0.5
        if isline:
            a1 += asin(min(self.r / l1, 1))
            a2 -= asin(min(self.r / l2, 1))
        arc = Arc(a1, a2)
        layer = 0 if isline else line.layer + 1
        return Path(self, line, arc, (l1+l2)/2, layer)

class Table:
    def __init__(self, loc, size, back, balls, tp='black8'):
        self.loc, self.size = loc, size
        self.back, self.balls = back, balls
        self.balls = [Ball(y,x,r,int(tp)) for y, x, r, tp in balls]
        self.unit = self.size[1]/100
        self.paths = self.hitpts = []
        if tp=='black8': self.make_pocket(1.5, 2.7)
        if tp=='snooker': self.make_pocket(1, 3)

    def make_pocket(self, k1, k2):
        unit, h, w = self.unit, *self.size
        mar = self.mar = unit * k1
        lst = []
        lst.append(Line(mar, mar+mar*k2, mar+mar*k2, mar))
        lst.append(Line(mar, w/2+mar*k2, mar, w/2-mar*k2))
        lst.append(Line(mar+mar*k2, w-mar, mar, w-mar-mar*k2))
        lst.append(Line(h-mar, w-mar-mar*k2, h-mar-mar*k2, w-mar))
        lst.append(Line(h-mar, w/2-mar*k2, h-mar, w/2+mar*k2))
        lst.append(Line(h-mar-mar*k2, mar, h-mar, mar+mar*k2))
        self.pocket = lst
        
    def show(self):
        import numpy as np
        plt.imshow(self.back)
        angs = np.linspace(0, np.pi*2, 36)
        rs, cs = np.cos(angs), np.sin(angs)
        lut = np.array([(255,255,255), (255,0,0),
            (255,255,0), (0,255,0), (128,0,0),
            (0,0,255), (255,128,128), (50,50,50)])/255
        for i in self.balls:
            plt.plot(cs*i.r+i.y, rs*i.r+i.x, c=lut[i.tp])
        h, w, mar = *self.size, self.mar
        plt.plot([mar, mar, w-mar, w-mar, mar],
                 [mar, h-mar, h-mar, mar, mar], 'blue')
        for line in self.pocket:
            r1, c1, r2, c2 = line.x1, line.y1, line.x2, line.y2
            plt.plot([c1,c2], [r1,r2], 'white')
        for p in self.paths: self.plot_path(p)
        if len(self.hitpts)>0:
            plt.plot(self.hitpts[:,1], self.hitpts[:,0], 'r.')
        plt.show()

    def plot_path(self, path):
        x, y, arc, l = path.frm.x, path.frm.y, path.arc, path.l
        p = path.get_point()
        #p3 = cos(arc.a2)*l+x, sin(arc.a2)*l+y
        plt.plot([p[1], y], [p[0], x], 'yellow')

    def solve(self, goal=-1, maxiter=1):
        self.paths = self.pocket.copy()
        for i in range(1000):
            if i == len(self.paths): break
            cur = self.paths[i]
            if i>5 and cur.layer == maxiter: break
            if isinstance(cur, Line): # 入袋球
                if goal!=-1: balls = [i for i in self.balls if i.tp==goal]
                else: balls = [i for i in self.balls if i.tp!=0]
            elif cur.layer == maxiter-2 and goal!=-1:
                balls = [i for i in self.balls if i.tp in (goal,0)]
            elif cur.layer == maxiter-1: # 回归母球
                balls = [i for i in self.balls if i.tp==0]
            else: balls = self.balls

            for ball in balls:
                if ball.tp==0 and goal!=-1 and cur.frm.tp!=goal: continue
                path = ball.to_line(cur)
                for b in self.balls:
                    if b != ball: path.shadow(b)
                for b in self.balls:
                    if b != ball and i>5 and b != path.to.frm:
                        path.inpoint(b)
                if path.arc.value()>0.0005:
                    self.paths.append(path)
        self.paths = self.paths[6:]

        for cur in self.paths[::-1]:
            if cur.frm.tp==0 and cur.layer>0:
                while cur.layer>0:
                    cur.layer *= -1
                    cur = cur.to
                    if cur.layer==0:
                        cur.layer=-10
        self.paths = [i for i in self.paths if i.layer<0]
        pts = [i for i in self.paths if i.frm.tp==0]
        rst = []
        for i in pts:
            x = cos(i.arc.mid()) * i.l + i.frm.x
            y = sin(i.arc.mid()) * i.l + i.frm.y
            # x, y, type, time, angle
            rst.append([x, y, i.to.frm.tp, abs(i.layer), min(i.to.arc.value()*1000, 100)])
        self.hitpts = np.array(rst)

if __name__ == '__main__':
    img = imread('https://user-images.githubusercontent.com/24822467/93710301-23978000-fb78-11ea-9908-eac1c8f8ae19.png')[:,:,:3]
    from extract import extract_table
    table = extract_table(img, 'snooker')
    if isinstance(table, tuple):
        table = Table(* table, 'snooker')
        table.solve(1, 1)
        start = time()
        #table.test()
        print(time()-start)
        table.show()
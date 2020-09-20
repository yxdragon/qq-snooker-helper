from PIL import Image
from skimage.io import imread
from skimage import color
from time import time
import numpy as np
from numpy.linalg import norm
import scipy.ndimage as ndimg
import matplotlib.pyplot as plt

# RGB转换HSV空间
def rgb2hsv(rgb):
    hsv = np.zeros(rgb.shape, dtype=np.float32)
    cmax = rgb.max(axis=-1)
    crng = rgb.ptp(axis=-1)
    np.clip(cmax, 1, 255, out=hsv[:,:,1])
    np.divide(crng, hsv[:,:,1], out=hsv[:,:,1])
    np.divide(cmax, 255, out=hsv[:,:,2])
    maxidx = np.argmax(rgb, axis=-1).ravel()
    colrgb = rgb.reshape(-1,3)
    idx = np.arange(colrgb.shape[0])
    lut = np.array([[1,2,0],[2,0,1]], dtype=np.uint8)
    h = (colrgb[idx, lut[0][maxidx]]).astype(np.float32)
    h -= colrgb[idx, lut[1][maxidx]]
    h[h==0] = 1
    np.clip(crng, 1, 255, out=crng)
    h /= crng.ravel()
    h += np.array([0,2,4], dtype=np.uint8)[maxidx]
    h /= 6; h %= 1
    hsv[:,:,0] = h.reshape(hsv.shape[:2])
    return hsv

# 制作HSV索引表
def make_lut():
    arr = np.mgrid[0:256,0:256,0:256].reshape(3,-1).T
    arr = arr.astype(np.uint8)
    lut = rgb2hsv(arr.reshape(1,-1,3))
    lut = (lut[0,:,0]*255).astype(np.uint8)
    return lut.reshape(256,256,256)

# 利用索引进行RGB到HSV转换
def rgb2hsv_lut(rgb, lut=[None]):
    if lut[0] is None: lut[0] = make_lut()
    r,g,b = rgb.reshape(-1,3).T
    return lut[0][r,g,b].reshape(rgb.shape[:2])
    
# 计算角度
def angleX(v):
    a = np.arccos(v[:,0] / (norm(v[:,:2], axis=1)+1e-5))
    return np.where(v[:,1]>=0,a ,np.pi * 2 - a)

# 精确定位, 根据圆心和采样点，组建法方程，进行最小二乘估计
def exactly(O, r, pts):
    n = len(pts)
    B = np.zeros((n*2, n+3))
    L = np.zeros(n*2)
    appro = np.zeros(n+3)
    appro[:n] = angleX(pts-O)
    appro[n:] = [O[0], O[1], r]
    try:
        for i in range(2): # 两次迭代，确保达到稳定
            L[::2] = appro[n]+appro[-1]*np.cos(appro[:n])-pts[:,0]
            L[1::2] = appro[n+1]+appro[-1]*np.sin(appro[:n])-pts[:,1]
            B[range(0,n*2,2),range(n)] = -appro[-1]*np.sin(appro[:n])
            B[range(1,n*2,2),range(n)] = appro[-1]*np.cos(appro[:n])
            B[::2,n],B[1::2,n+1] = 1, 1
            B[::2,-1] = np.cos(appro[:n])
            B[1::2,-1] = np.sin(appro[:n])
            NN = np.linalg.inv(np.dot(B.T,B))
            x = np.dot(NN, np.dot(B.T,L))
            v = np.dot(B,x)-L
            appro -= x
    except:
        print(O, r, pts)
    if not(appro[-1]>5 and appro[-1]<50): 
        return (None, None), None
    return appro[[-3,-2]], appro[-1]

#a = np.arccos(v[:,0] / norm(v[:,:2], axis=1))
# 查找背景
def find_ground(img, tor=5):
    r, c = np.array(img.shape[:2])//2
    center = img[r-100:r+100, c-100:c+100]
    back = np.argmax(np.bincount(center.ravel()))
    msk = np.abs(img.astype(np.int16) - back)<tor
    lab, n = ndimg.label(msk)
    hist = np.bincount(lab.ravel())
    if hist[1:].max() < 1e4: return None
    if np.argmax(hist[1:])==0: return None
    msk = lab == np.argmax(hist[1:]) + 1
    sr, sc = ndimg.find_objects(msk)[0]
    loc = sr.start, sc.start
    size = sr.stop - loc[0], sc.stop - loc[1]
    return loc, size, sr, sc, msk[sr, sc]

# 查找一个球
def find_one(img, cs, r=18, a=20):
    h, w = img.shape
    if cs[0]<r+1 or cs[1]<r+1 or cs[0]>h-r-1 or cs[1]>w-r-1:
        return (None, None), None
    rs, pts = np.arange(r), []
    for i in np.linspace(0, np.pi*2, a, endpoint=False):
        rcs = rs[:,None] * (np.cos(i), np.sin(i)) + cs
        rcs = rcs.round().astype(int).T
        ns = rs[img[rcs[0], rcs[1]]]
        if len(ns)==0: continue
        pts.append(rcs.T[ns.min()])
    if len(pts)<10: return (None, None), None
    return exactly(cs, r, np.array(pts))

# 检测球
def find_ball(img):
    dist = ndimg.binary_dilation(img, np.ones((13, 13)))
    dist[:,[0,-1]] = 0; dist[[0,-1],:] = 0
    lab, n = ndimg.label(~dist)
    objs = ndimg.find_objects(lab)[1:]
    cs = [(i.start+i.stop, j.start+j.stop) for i,j in objs]
    balls = []
    for i in np.array(cs)/2:
        (r, c), ra = find_one(img, i)
        if not ra is None: balls.append([r, c, ra])
    if len(balls)==0: return balls
    balls = np.array(balls)
    balls[:,2] = balls[:,2].mean()-0.5
    return balls

# 提取颜色
def detect_color(img, balls, mode='snooker'):
    r = int(balls[0,2]) - 1
    rcs = np.mgrid[-r:r+1, -r:r+1].reshape(2,-1).T
    rcs = rcs[norm(rcs, axis=1) < r]
    colors = []
    for r,c in balls[:,:2]:
        rs, cs = (rcs + (int(r), int(c))).T
        colors.append(img[rs, cs])
    colors = np.array(colors).astype(np.int16)
    colors = np.sort(colors, axis=1)
    colors = colors[:,len(rcs)//4:-len(rcs)//4]
    if mode=='snooker':
        snklut = [21, 0, 34, 73, 12, 171, 221, 42]
        cs = [np.argmax(np.bincount(i)) for i in colors]
        diff = np.abs(np.array(cs)[:,None] - snklut)
        return np.argmin(diff, axis=-1)
    
    if mode=='black8':
        bins = np.array([np.bincount(i, minlength=256) for i in colors])
        mean = np.argmax(bins, axis=-1)
        std = (np.std(colors, axis=1)>1) + 1
        std[(std==1) & (np.abs(mean-42)<3)] = 7
        n = (np.abs(colors-28)<3).sum(axis=1)
        n = bins[:,25:30].max(axis=1)
        #print(mean)
        #print(np.bincount(colors[5]))
        #print(np.bincount(colors[9]))
        std[np.argmax(n)] = 0
        return std

# lut = np.load('lut.npy')
# 提取球桌信息
def extract_table(img, mode='snooker'):
    #hsv = (rgb2hsv(img[:,:,:3]) * 255).astype(np.uint8)
    hsv = rgb2hsv_lut(img)
    ground = find_ground(hsv)
    if ground is None: return '未检测到球桌，请勿遮挡'
    loc, size, sr, sc, back = ground
    balls = find_ball(back)
    if len(balls)==0: return '全部球已入袋'
    tps = detect_color(hsv[sr, sc], balls, mode)
    balls = np.hstack((balls, tps[:,None]))
    return loc, size, img[sr, sc], balls
    
if __name__ == '__main__':
    img = imread('https://user-images.githubusercontent.com/24822467/93710301-23978000-fb78-11ea-9908-eac1c8f8ae19.png')[:,:,:3]
    start = time()
    #hsv = (rgb2hsv(img[:,:,:0]) * 255).astype(np.uint8)
    ax = plt.subplot(221)
    ax.imshow(img)

    hsv = rgb2hsv_lut(img)
    print('to hsv', time()-start)
    ax = plt.subplot(222)
    ax.imshow(hsv)

    start = time()
    loc, size, sr, sc, back = find_ground(hsv)
    print('back', time()-start)
    ax = plt.subplot(223)
    ax.imshow(back)

    start = time()
    balls = find_ball(back)

    ax = plt.subplot(224)
    ax.imshow(img[sr, sc])
    ax.plot(balls[:,1], balls[:,0], 'r.')

    plt.show()

    print('ball', time()-start)

    start = time()
    tps = detect_color(hsv[sr, sc], balls)
    print('detect', time()-start)
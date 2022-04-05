
import os, pickle, gzip
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt


f = gzip.open(r'C:\Users\i_hat\Desktop\losable\mnist.pkl.gz', 'rb')
_tr, _va, _te = pickle.load(f, encoding = "latin1")
f.close()
tr_x = [np.reshape(x, (784, 1)) for x in _tr[0]] # reshape x's
va_x = [np.reshape(x, (784, 1)) for x in _va[0]]
te_x = [np.reshape(x, (784, 1)) for x in _te[0]]
va_data = list(zip(va_x, _va[1])) # list of tuples of (x,y)
te_data = list(zip(te_x, _te[1]))
x_orig = np.array(tr_x).squeeze() * 255.0
x_orig.shape

siz                 = 28     # picture length and width
SAVE_DIR            = r'C:\Users\pwnag\Desktop\sup\deep_larn\anime_PCA'
IMG_DIR             = r'C:\Users\pwnag\Desktop\sup\deep_larn\anime_face_400\images'


def plot_some_pics(_data):
    n                   = 5
    pic                 = np.zeros((siz * n, siz * n))
    for i in range(n):
        for j in range(n):
            pic[i * siz : (i + 1) * siz, j * siz : (j + 1) * siz] = _data[j + n * i,:].reshape(siz, siz)
    _                   = plt.figure(figsize=(8, 8))
    _                   = plt.imshow(pic) # fchollet uses jet. rainbow, gist_rainbow, hsv, 
    _                   = plt.axis('off')
    plt.show()


def eig(S):
    va, vc          = np.linalg.eigh(S)  
    _sorted         = np.argsort(-va) # sorting them in decrasing order
    va              = va[_sorted]
    vc              = vc[:, _sorted]
    return (va, vc)


#if False: # this part generates mu,std,vecs for the entire data set which can take 10 mins
plot_some_pics(x_orig)


# standardize & do eigen
mu                  = np.mean(x_orig, axis = 0, keepdims = True)
_x                  = x_orig - mu                          # subtract mean (center data at 0)
_std                = np.std(_x, axis = 0, keepdims = True)# calculate variance of features
std                 = _std.copy()                          # fix zero variances
std[std == 0.0]     = 1.0
x                   = _x / std                             # scale the data
my_cov              = (x.T @ x) / (x.shape[1] - 1)         # https://datascienceplus.com/understanding-the-covariance-matrix/
vals, vecs          = eig(my_cov)                          # eigenvecs of cov matrix 



# do pca
components          = 200
U                   = vecs[:, range(components)]       # take a subset


def unstand(x, add_mean = True):
    if add_mean:
        return x * std + mu
    return x * std


def to_uint8(x): # prepare picture for display. i min-max scaled it. not sure if this is good but it gives cool visuals. 
    #x = x - np.min(x)
    #return 255.0 * x / np.max(x) 
    x = np.clip(x,0,255)
    return x.astype(np.uint8)



class Soy():
    def _reset(self):
        [self.vals[i].set(0.0) for i in range(len(self.vals))]
        self.refresh(0)

    def _change_mean_option(self):
        self.add_mean = not self.add_mean
        self.refresh(0)

    def reconstruct(self):
        self.proj = np.array([[i.get() for i in self.vals]])
        self.x = to_uint8(unstand(self.proj @ U.T, self.add_mean))

    def refresh(self, e):
        self.reconstruct()
        self.xdata                              = b'P5 28 28 255 ' + self.x.tobytes()
        self.photo                              = tk.PhotoImage(width=siz, height=siz, data=self.xdata, format = 'PPM').zoom(16)
        #self.p = self.x.reshape([siz,siz])
        #self.photo = ImageTk.PhotoImage(image = Image.fromarray(self.p)) # https://stackoverflow.com/questions/58411250/photoimage-zoom
        #self.photo = self.photo._PhotoImage__photo.zoom(8)
        self.canvas_area.create_image(0,0,image =self.photo,anchor=tk.NW)
        self.canvas_area.update()

    def start(self):
        self.add_mean                               = True
        root                                        = tk.Tk()
        menu_left                                   = tk.Canvas(root, width=150, height = 400, bg = 'black')
        menu_left.grid(row                          = 0, column=0, sticky = 'nsew')
        sf                                          = ttk.Frame(menu_left)
        sf.bind("<Configure>",   lambda e: menu_left.configure(scrollregion = menu_left.bbox("all")))
        root.bind('<Up>'     ,   lambda x: menu_left.yview_scroll(-10, "units"))
        root.bind('<Down>'   ,   lambda x: menu_left.yview_scroll(10, "units")) 
        root.bind("<Escape>" ,   lambda x: root.destroy())
        root.bind('r',           lambda x: self._reset())
        root.bind('a',           lambda x: self._change_mean_option())
        menu_left.create_window((0, 0), window      =sf, anchor="nw")

        self.vals                                   = [tk.DoubleVar() for i in range(components)]
        labs                                        = [ttk.Label(sf, text=f"{i}") for i in range(components)]
        slds                                        = [None for i in range(components)]
        for i in range(components):
            slds[i]                                 = ttk.Scale(sf, from_ = -50, to = 50, orient = 'horizontal', variable = self.vals[i], command = self.refresh)
            slds[i].grid(column                     = 1, row = i, columnspan = 1, sticky = 'nsew')
            labs[i].grid(column                     = 0, row = i, columnspan = 1, sticky = 'nsew')

        self.canvas_area                            = tk.Canvas(root, width=540, height=540, bg = 'black')
        self.canvas_area.grid(row                   =0, column=1, sticky = 'nsew') 
        root.grid_rowconfigure(1, weight            =1)
        root.grid_columnconfigure(1, weight         =1)
        self.refresh(0)
        root.mainloop()

soy                                                 = Soy()
soy.start()
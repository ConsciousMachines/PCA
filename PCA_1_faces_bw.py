
# we want to explore the latent space
# the latent space is theoretically N(0,1) - PCA assumes the dependencies are linear (and Gaussian + linear dep = independent)
# we travel along its axes and reconstruct the points. 

# we want to explore what features the eigenvector controls
# this is the same as traveling along an axis in latent space. 
# it corresponds to multiplying that eigenvector by the axis's value during reconstruction. 

# thus we can either travel in latent space starting at point 0 and walking up axes
# or we can start at a known face (its encoding) and traveling along axes corresponds to traveling along corresponding eigenvector

# CODE SOurCES
# video :D 
# https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python
# slider
# https://www.pythontutorial.net/tkinter/tkinter-slider/
# actual app with areas
# https://stackoverflow.com/questions/36506152/tkinter-grid-or-pack-inside-a-grid
# display np array
# https://python-forum.io/thread-528.html
# scrollable frame + scroll using key and xview_scroll(10, "units")
# https://blog.teclado.com/tkinter-scrollable-frames/
# https://stackoverflow.com/questions/46194948/what-are-the-tkinter-events-for-horizontal-edge-scrolling-in-linux/46194949#46194949
# apparently you can draw arrays using create_line ? 
# https://stackoverflow.com/questions/4084383/python-tkinter-stretch-an-image-horizontally
# https://stackoverflow.com/questions/26178869/is-it-possible-to-apply-gradient-colours-to-bg-of-tkinter-python-widgets


# https://www.kaggle.com/code/pranjallk1995/pca-from-scratch-for-image-reconstruction/notebook
# this code is actually from:
# https://zhangruochi.com/Principal-Component-Analysis/2019/09/20/


import os, pickle
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk


siz                 = 64     # picture length and width
SAVE_DIR            = r'C:\Users\i_hat\Desktop\bastl\py\deep_larn\anime_PCA'
IMG_DIR             = r'C:\Users\i_hat\Desktop\losable\anime_face_400\images'


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


if False: # this part generates mu,std,vecs for the entire data set which can take 10 mins
        
    # Loading data and reducing size to 64 x 64 pixels
    _samples            = len(os.listdir(IMG_DIR))
    x_orig              = np.zeros([_samples, siz * siz])
    for i, img in enumerate(os.listdir(IMG_DIR)):
        _img            = Image.open(os.path.join(IMG_DIR,img)) 
        x_orig[i,:]     = np.array(_img.resize((siz, siz), Image.ANTIALIAS)).mean(axis = 2).flatten()
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


    params = [mu, std, vecs]
    #pickle.dump(params, open(os.path.join(SAVE_DIR, 'anime_400_PCA_1.pkl'), 'wb'))

    
params = pickle.load(open(os.path.join(SAVE_DIR, 'anime_400_PCA_1.pkl'), 'rb'))
mu, std, vecs = params


# do pca
components          = 200
U                   = vecs[:, range(components)]       # take a subset


def from_latent(z, add_mean = True):
    x = z @ U.T
    if add_mean:
        return x * std + mu
    return x * std


class Viewer():
    def reconstruct(self):
        code = np.array([[i.get() for i in self.vals]])
        self.x = np.clip(from_latent(code, self.add_mean), 0.0, 255.0).astype(np.uint8)

    def refresh(self, e):
        self.reconstruct()
        self.photo                                       = tk.PhotoImage(width=siz, height=siz, data=b'P5 64 64 255 ' + self.x.tobytes(), format = 'PPM').zoom(8)
        self.canvas_area.create_image(0,0,image          = self.photo,anchor=tk.NW)
        self.canvas_area.update()

    def key_press(self, e):
        if e.char                                        == 'r':      # reset sliders to 0
            [self.vals[i].set(0.0) for i in range(len(self.vals))]
        elif e.char                                      == 'a':      # toggle mean option 
            self.add_mean                                = not self.add_mean
        elif e.char                                      == 't':      # randomize
            num_feats                                    = min(self.num_sliders, 50) # dont do all the features as most are noise
            rand                                         = np.clip(np.random.randn(num_feats) * 2.0, -50.0, 50.0)
            for i in range(num_feats):
                self.vals[i].set(rand[i])
        elif e.char                                      == 'e':      # remember the encoding 
            self.remember_vecs.append(np.array([[i.get() for i in self.vals]]))
        elif e.keysym                                    == 'Escape': # quit 
            return self.root.destroy()
        elif e.keysym                                    == 'Down':   # scroll down in slider menu
            self.menu_left.yview_scroll(10, "units")
        elif e.keysym                                    == 'Up':     # scroll up in slider menu
            self.menu_left.yview_scroll(-10, "units")
        else:
            print(e)
        self.refresh(0)

    def start(self, num_sliders                          = 100):
        self.root                                        = tk.Tk()
        self.menu_left                                   = tk.Canvas(self.root, width=150, height = 400, bg = 'black')
        self.menu_left.grid(row                          = 0, column=0, sticky = 'nsew')
        sf                                               = ttk.Frame(self.menu_left)
        sf.bind("<Configure>",  lambda e: self.menu_left.configure(scrollregion = self.menu_left.bbox("all")))
        self.root.bind('<Key>', lambda x: self.key_press(x))
        self.menu_left.create_window((0, 0), window      =sf, anchor="nw")

        self.remember_vecs                               = []
        self.num_sliders                                 = num_sliders
        self.add_mean                                    = True
        self.vals                                        = [tk.DoubleVar() for i in range(self.num_sliders)]
        labs                                             = [ttk.Label(sf, text=f"{i}") for i in range(self.num_sliders)]
        slds                                             = [None for i in range(self.num_sliders)]
        for i in range(self.num_sliders):
            slds[i]                                      = ttk.Scale(sf, from_ = -50.0, to = 50.0, orient = 'horizontal', variable = self.vals[i], command = self.refresh)
            slds[i].grid(column                          = 1, row = i, columnspan = 1, sticky = 'nsew')
            labs[i].grid(column                          = 0, row = i, columnspan = 1, sticky = 'nsew')

        self.canvas_area                                 = tk.Canvas(self.root, width=540, height=540, bg = 'black')
        self.canvas_area.grid(row                        = 0, column=1, sticky = 'nsew') 
        self.root.grid_rowconfigure(1, weight            = 1)
        self.root.grid_columnconfigure(1, weight         = 1)
        self.refresh(0)
        self.root.mainloop()
        

v                                                        = Viewer()
v.start(components)



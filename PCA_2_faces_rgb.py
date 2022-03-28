
import os, pickle
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk


siz                 = 64     # picture length and width
SAVE_DIR            = r'C:\Users\pwnag\Desktop\sup\deep_larn\anime_PCA'
IMG_DIR             = r'C:\Users\pwnag\Desktop\sup\deep_larn\anime_face_400\images'


def plot_some_pics(d1, d2, d3):
    n                   = 5
    # channels need to be interleaved
    _data = np.concatenate([np.expand_dims(d1[:n*n,:],2),np.expand_dims(d2[:n*n,:],2),np.expand_dims(d3[:n*n,:],2)],axis=2)
    pic                 = np.zeros((siz * n, siz * n, 3), dtype = np.uint8)
    for i in range(n):
        for j in range(n):
            _x = _data[j + n * i,:,:].reshape([siz,siz,3])
            _x = _x - np.min(_x)
            _x = (255.0 * _x / np.max(_x)).astype(np.uint8)
            pic[i * siz : (i + 1) * siz, j * siz : (j + 1) * siz, :] = _x.reshape([siz,siz,3])
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


def do_pca(m):
    cov_mat = (m.T @ m) / (m.shape[1] - 1) # https://datascienceplus.com/understanding-the-covariance-matrix/
    return eig(cov_mat)


if False:
    # Loading data and reducing size to 64 x 64 pixels
    __i = 1
    _samples            = 20000
    x_orig_r            = np.zeros([_samples, siz * siz])
    x_orig_g            = np.zeros([_samples, siz * siz])
    x_orig_b            = np.zeros([_samples, siz * siz])
    for i, img in enumerate(os.listdir(IMG_DIR)[__i*_samples:(__i + 1) * _samples]):
        _img            = Image.open(os.path.join(IMG_DIR,img))
        assert _img.mode == 'RGB'
        _data = np.array(_img.resize((siz, siz), Image.ANTIALIAS))
        x_orig_r[i,:]     = _data[:,:,0].flatten()
        x_orig_g[i,:]     = _data[:,:,1].flatten()
        x_orig_b[i,:]     = _data[:,:,2].flatten()
    #plot_some_pics(x_orig_r, x_orig_g, x_orig_b)


    # standardize & do eigen
    def standardize(inp):
        mu                  = np.mean(inp, axis = 0, keepdims = True)
        _x                  = inp - mu                          # subtract mean (center data at 0)
        _std                = np.std(_x, axis = 0, keepdims = True)# calculate variance of features
        std                 = _std.copy()                          # fix zero variances
        std[std == 0.0]     = 1.0
        x                   = _x / std                             # scale the data
        return x, mu, std
    xr, mur, stdr = standardize(x_orig_r)
    xg, mug, stdg = standardize(x_orig_g)
    xb, mub, stdb = standardize(x_orig_b)
    # get covariance per channel
    _, vecsr = do_pca(xr)
    _, vecsg = do_pca(xg)
    _, vecsb = do_pca(xb)




params = pickle.load(open(os.path.join(SAVE_DIR, r'anime_400_PCA_2_miku.pkl'), 'rb'))
vecsr, vecsg, vecsb, mur, stdr, mug, stdg, mub, stdb = params

# do pca
components          = 200
UR                  = vecsr[:, range(components)]       
UG                  = vecsg[:, range(components)]       
UB                  = vecsb[:, range(components)]       
#recon_r                   = (xr @ UR @ UR.T) * stdr + mur
#recon_g                   = (xg @ UG @ UG.T) * stdg + mug
#recon_b                   = (xb @ UB @ UB.T) * stdb + mub
#plot_some_pics(recon_r, recon_g, recon_b)

#def get_face(i): # returns the i^th face from original data
#    return (x_orig_r[[i],:], x_orig_g[[i],:], x_orig_b[[i],:])

#def stand(x): # standardize original data by removing its mean and dividing by std
#    x1, x2, x3 = x 
#    return ((x1 - mur)/stdr, (x2-mug)/stdg, (x3-mub)/stdb)

#def project(z): # projects to the latent space
#    z1, z2, z3 = z
#    return (z1 @ UR, z2 @ UG, z3 @ UB)

#def show_face(f): # display one face, since the convention is to use a tuple of the R,G,B matrices
#    d1, d2, d3 = f
#    _data = np.concatenate([np.expand_dims(d1[:,:],2),np.expand_dims(d2[:,:],2),np.expand_dims(d3[:,:],2)],axis=2).astype(np.uint8)
#    plt.imshow(_data.reshape([siz,siz,3]))
#    plt.show()

def unproject(z): # comes back from latent space 
    z1, z2, z3 = z
    return (z1 @ UR.T, z2 @ UG.T, z3 @ UB.T)

def unstand(x, add_mean = True): # when we transpose from the latent space, we need to unstandardize it
    x1, x2, x3 = x 
    if add_mean:
        return (x1 * stdr + mur, x2 * stdg + mug, x3 * stdb + mub)
    return (x1 * stdr, x2 * stdg, x3 * stdb)

def to_uint8(x): # prepare picture for display. i min-max scaled it. not sure if this is good but it gives cool visuals. 
    x = x - np.min(x)
    return (255.0 * x / np.max(x)).astype(np.uint8)


#one_face = get_face(3)
#show_face(one_face)

#one_face = stand(one_face)
#proj = project(one_face)

#unproj = unproject(proj)
#recon = unstand(unproj)
#show_face([to_uint8(i) for i in recon])







class Soy():
    def _reset(self):
        [self.vals[i].set(0.0) for i in range(len(self.vals))]
        self.refresh(0)

    def _change_mean_option(self):
        self.add_mean = not self.add_mean
        self.refresh(0)

    def reconstruct(self):
        self.proj = np.array([[i.get() for i in self.vals]])
        self.x = [to_uint8(i) for i in unstand(unproject((self.proj,self.proj,self.proj)), self.add_mean)]

    def refresh(self, e):
        self.reconstruct()
        self.p = np.concatenate([np.expand_dims(self.x[0][:,:],2),np.expand_dims(self.x[1][:,:],2),np.expand_dims(self.x[2][:,:],2)],axis=2).squeeze().reshape([siz,siz,3])
        self.photo = ImageTk.PhotoImage(image = Image.fromarray(self.p)) # https://stackoverflow.com/questions/58411250/photoimage-zoom
        self.photo = self.photo._PhotoImage__photo.zoom(8)
        self.canvas_area.create_image(0,0,image = self.photo, anchor=tk.NW)
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




# IMPORTANT SAVE # 1 - discovered Hatsune Miku feature at slider 33, and she friggin rotates at feature 34. 
#params = [vecsr, vecsg, vecsb, mur, stdr, mug, stdg, mub, stdb]
#pickle.dump(params, open(r'C:\Users\pwnag\Desktop\sup\deep_larn\anime_PCA\anime_400_PCA_2_miku.pkl', 'wb'))


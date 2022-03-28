
import numpy as np
import tkinter as tk
from tkinter import ttk
import zipfile, io, os, pickle
from PIL import Image, ImageTk


siz                 = 64     # picture length and width
IMG_DIR             = r'C:\Users\pwnag\Desktop\sup\deep_larn\anime_face_400\images'
SAVE_DIR            = r'C:\Users\pwnag\Desktop\sup\deep_larn\anime_PCA'


def generate_chunk_inds(total, sample_size): # generate indices for chunks of data 
    _inds = list(range(0, total, sample_size)) + [total]
    inds = [(_inds[i], _inds[i+1]) for i in range(len(_inds)-1)]
    return inds


def get_data_chunk(_start, _end, siz = siz, IMG_DIR = IMG_DIR): # retrieve data between a range of indices
    _samples              = _end - _start
    x_orig_r              = np.zeros([_samples, siz * siz])
    x_orig_g              = np.zeros([_samples, siz * siz])
    x_orig_b              = np.zeros([_samples, siz * siz])
    for i, img in enumerate(os.listdir(IMG_DIR)[_start: _end]):
        _img              = Image.open(os.path.join(IMG_DIR, img))
        assert _img.mode  == 'RGB'
        _data             = np.array(_img.resize((siz, siz), Image.ANTIALIAS))
        x_orig_r[i,:]     = _data[:,:,0].flatten()
        x_orig_g[i,:]     = _data[:,:,1].flatten()
        x_orig_b[i,:]     = _data[:,:,2].flatten()
    return x_orig_r, x_orig_g, x_orig_b


def get_chunk_cov(x, mu, std, n):
    _x = (x - mu) / std
    return _x.T @ _x / n


def eig(S):
    va, vc          = np.linalg.eigh(S)  
    _sorted         = np.argsort(-va) # sorting them in decrasing order
    va              = va[_sorted]
    vc              = vc[:, _sorted]
    return (va, vc)


if False:
    # we already gathered the means and the stds before. 
    vecsr, vecsg, vecsb, mu_r, mu_g, mu_b, std_r, std_g, std_b = pickle.load(open(os.path.join(SAVE_DIR, 'anime_400_PCA_3.pkl'), 'rb'))
    mu                                                         = np.concatenate([mu_r, mu_g, mu_b], 1)
    std                                                        = np.concatenate([std_r, std_g, std_b], 1)

    # S T E P   3   :   C O V A R I A N C E 
    inds           = generate_chunk_inds(len(os.listdir(IMG_DIR)), 10_000)
    my_cov         = np.zeros([siz*siz*3, siz*siz*3])
    for start, end in inds:
        data_chunk = get_data_chunk(start, end)
        x          = np.concatenate([data_chunk[0], data_chunk[1], data_chunk[2]], 1)
        my_cov    += get_chunk_cov(x, mu, std, inds[-1][-1])

    # S T E P   4   :   P C A
    _, vecs = eig(my_cov)
    #pickle.dump([mu, std, vecs], open(os.path.join(SAVE_DIR, 'anime_400_PCA_4.pkl'), 'wb'))
    #pickle.dump(my_cov, open(os.path.join(SAVE_DIR, 'anime_400_PCA_4_my_cov.pkl'), 'wb'))

    #==================== start experimetn - apparently i did it correct this time. 
    #my_cov = pickle.load(open(os.path.join(SAVE_DIR, 'anime_400_PCA_4_my_cov.pkl'), 'rb'))
    #m1 = np.ones([siz*siz, siz*siz])
    #m0 = np.zeros([siz*siz, siz*siz])
    #my_cov = my_cov * np.concatenate([
    #    np.concatenate([m1,m0,m0]),
    #    np.concatenate([m0,m1,m0]),
    #    np.concatenate([m0,m0,m1]),
    #], 1)
    #_, vecs = eig(my_cov)
    #==================== end experimetn


mu, std, vecs = pickle.load(open(os.path.join(SAVE_DIR, 'anime_400_PCA_4.pkl'), 'rb'))
components          = 200
U                   = vecs[:, range(components)]       


def unstand(x, add_mean = True):
    if add_mean:
        return x * std + mu
    return x * std

def to_uint8(x): # prepare picture for display. i min-max scaled it. not sure if this is good but it gives cool visuals. 
    #x = x - np.min(x)
    #x = 255.0 * x / np.max(x)
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
        self.x = to_uint8(unstand(self.proj @ U.T, self.add_mean)).reshape([3,siz,siz])

    def refresh(self, e):
        self.reconstruct()
        self.p = np.concatenate([np.expand_dims(self.x[0,:,:],2),np.expand_dims(self.x[1,:,:],2),np.expand_dims(self.x[2,:,:],2)],axis=2).squeeze().reshape([siz,siz,3])
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




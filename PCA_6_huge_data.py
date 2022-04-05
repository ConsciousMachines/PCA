'''

import cupy as cp
cp.cuda.Device()

x_cpu = np.array([1, 2, 3])
x_gpu = cp.asarray(x_cpu)  # move the data to the current device.

with cp.cuda.Device(0):
    x_gpu_0 = cp.ndarray([1, 2, 3])  # create an array in GPU 0
x_cpu = cp.asnumpy(x_gpu)  # move the array to the host.
'''




import zipfile
import numpy as np
import io
import PIL.Image as im



def generate_chunk_inds(total, sample_size): # generate indices for chunks of data 
    _inds = list(range(0, total, sample_size)) + [total]
    inds = [(i, _inds[i], _inds[i+1]) for i in range(len(_inds)-1)]
    return inds


def get_sample_weights(inds): # get the weights of the samples: ni / n  
    n = inds[-1][-1]
    return [(end - start) / n for _, start, end in inds]


def get_data_chunk(start, end, files, siz):
    x                  = np.zeros([end - start, siz * siz * 3], dtype = np.uint8)
    for i in range(start, end):
        one_file       = files[i]
        orig_im        = im.open(io.BytesIO(zip.open(one_file).read()))
        d              = orig_im.size[0]
        crop_im        = orig_im.crop((d / 3, d / 3, 2 * d / 3, 2 * d / 3)).resize([siz,siz])
        x[i - start,:] = np.array(crop_im, dtype = np.uint8).flatten()
    return x


siz            = 128
path           = r'C:\Users\i_hat\Desktop\losable\anime_aligned.zip'
SAVE_DIR       = r'C:\Users\i_hat\Desktop\bastl\py\deep_larn\anime_PCA'

zip            = zipfile.ZipFile(path)
files          = zip.namelist()

_sample_size   = 10_000
inds           = generate_chunk_inds(len(files), _sample_size)
w              = get_sample_weights(inds)


# S T E P   1   :   G A T H E R   T H E   M E A N S

mean           = np.zeros([1, siz*siz*3])
for i, start, end in inds:
    x          = get_data_chunk(start, end, files, siz)
    mean      += w[i] * np.mean(x, 0, keepdims = True)








# S T E P   2   :   G A T H E R   T H E   S T D S

var             = np.zeros([1, siz*siz*3])
for i, start, end in inds:
    x           = get_data_chunk(start, end, files, siz)
    var        += w[i] * np.mean(np.square(x - mean), axis = 0, keepdims = True)
std             = np.sqrt(var)
std[std == 0.0] = 1.0







# S T E P   3   :   C O V A R I A N C E 

cov             = np.zeros([siz*siz*3, siz*siz*3])
for start, end in inds:
    x           = get_data_chunk(start, end, files, siz)
    _x          = (x - mean) / std # standardize the data using the mean and std
    cov        += _x.T @ _x / inds[-1][-1]





# S T E P   4   :   P C A
def eig(S):
    va, vc          = np.linalg.eigh(S)  
    _sorted         = np.argsort(-va) # sorting them in decrasing order
    va              = va[_sorted]
    vc              = vc[:, _sorted]
    return (va, vc)

_, vecs = eig(cov) # TODO: use cupy

pickle.dump([vecs, mean, std], open(os.path.join(SAVE_DIR, 'anime_100gb_PCA_6.pkl'), 'wb'))
















import numpy as np
import tkinter as tk
from tkinter import ttk
import zipfile, io, os, pickle
from PIL import Image, ImageTk


components = 200
vecsr, vecsg, vecsb, mu_r, mu_g, mu_b, std_r, std_g, std_b = pickle.load(open(os.path.join(SAVE_DIR, 'anime_400_PCA_3.pkl'), 'rb'))
UR                  = vecsr[:, range(components)]       
UG                  = vecsg[:, range(components)]       
UB                  = vecsb[:, range(components)] 
for i in range(components): # if v1 and v2 are opposite, then sign(dot(v1,v2))*v2 will face the same. 
    UG[:,i] *= np.sign(np.dot(UR[:,i], UG[:,i])) # we know the vecs have similar features but may point in diff directions.
    UB[:,i] *= np.sign(np.dot(UG[:,i], UB[:,i])) # so flip them if they differ (if dot = -1)


U                   = np.stack([UR.T, UG.T, UB.T]) # numpy multiplies an array of matrices as [n, m] @ [matrices, m, k]
std                 = np.stack([std_r, std_g, std_b]) # so we can do code : [1,200] ;; code @ U 
mu                  = np.stack([mu_r, mu_g, mu_b])



def from_latent(z, add_mean = True): 
    x = z @ U # comes back from latent space 
    if add_mean:
        return x * std + mu # when we transpose from the latent space, we need to unstandardize it
    return x * std


class Viewer():
    def reconstruct(self):
        code = np.array([[i.get() for i in self.vals]])
        self.x = np.clip(from_latent(code, self.add_mean), 0.0, 255.0).astype(np.uint8).transpose([1,2,0]).reshape([siz,siz,3]) 
        # WRONG BUT COOL: i think this visualizes the 3 vectors independently?
        #self.x = np.clip(from_latent(code, self.add_mean), 0.0, 255.0).astype(np.uint8).reshape([siz,siz,3]) 

    def refresh(self, e):
        self.reconstruct()
        self.photo = ImageTk.PhotoImage(image = Image.fromarray(self.x)) # https://stackoverflow.com/questions/58411250/photoimage-zoom
        self.photo = self.photo._PhotoImage__photo.zoom(8)
        self.canvas_area.create_image(0,0,image = self.photo, anchor=tk.NW)
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


v                                                 = Viewer()
v.start(components)

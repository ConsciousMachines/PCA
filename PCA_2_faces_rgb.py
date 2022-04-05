
import os, pickle
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk


siz                 = 64     # picture length and width
SAVE_DIR            = r'C:\Users\i_hat\Desktop\bastl\py\deep_larn\anime_PCA'
IMG_DIR             = r'C:\Users\i_hat\Desktop\losable\anime_face_400\images'


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

    # IMPORTANT SAVE # 1 - discovered Hatsune Miku feature at slider 33, and she friggin rotates at feature 34. 
    #pickle.dump([vecsr, vecsg, vecsb, mur, stdr, mug, stdg, mub, stdb], open(os.path.join(SAVE_DIR, r'anime_400_PCA_2_miku.pkl'), 'wb'))



vecsr, vecsg, vecsb, mur, stdr, mug, stdg, mub, stdb = pickle.load(open(os.path.join(SAVE_DIR, r'anime_400_PCA_2_miku.pkl'), 'rb'))

# do pca
components          = 200
UR                  = vecsr[:, range(components)]       
UG                  = vecsg[:, range(components)]       
UB                  = vecsb[:, range(components)]   
#for i in range(components): # if v1 and v2 are opposite, then sign(dot(v1,v2))*v2 will face the same. 
#    UG[:,i] = np.sign(np.dot(UR[:,i], UG[:,i])) * UG[:,i] # we know the vecs have similar features but may point in diff directions.
#    UB[:,i] = np.sign(np.dot(UG[:,i], UB[:,i])) * UB[:,i] # so flip them if they differ (if dot = -1)
# ^ this would be correct and i do it in PCA_3, but here i want the Hatsune Miku effect :p

U                   = np.stack([UR.T, UG.T, UB.T]) # numpy multiplies an array of matrices as [n, m] @ [matrices, m, k]
std                 = np.stack([stdr, stdg, stdb]) # so we can do code : [1,200] ;; code @ U 
mu                  = np.stack([mur, mug, mub])


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







import matplotlib.pyplot as plt
from matplotlib import animation


steps = 30
frames = np.zeros([steps * len(v.remember_vecs), components])

for i in range(len(v.remember_vecs)):
    v1 = v.remember_vecs[i]
    v2 = v.remember_vecs[(i+1) % len(v.remember_vecs)]
    dv = v2 - v1
    for j in range(steps):
        frames[i*steps + j,:] = v1 + dv * (float(j) / steps)

fig = plt.figure()
_ = plt.axis('off')
images = []
for c in range(frames.shape[0]):
    morphed = np.clip(from_latent(frames[[c],:]),0,255).astype(np.uint8).reshape([3,siz,siz]).transpose([1,2,0])
    images.append([plt.imshow(morphed, animated=True)])

ani = animation.ArtistAnimation(fig, images, interval=30)
ani.save(os.path.join(SAVE_DIR, 'PCA_2_miku.mp4'), writer = animation.FFMpegWriter(fps = 60))

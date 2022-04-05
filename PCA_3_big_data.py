

# B I G   D A D A -----------------------------------------------------------------------------------------------
# B I G   D A D A -----------------------------------------------------------------------------------------------


# i think that if we have n samples and m features, and n is large, our cov matrix is m x m, 
# and the process can be split up into chunks where we take b = n / 10 at a time.
# for example, the first entry is, for i in [1..n]:
# var(x1) = 1/n sum_1_n (x1i - x1bar)^2
# similarly, 
# cov(x1, x2) = 1/n sum_1_n (x1i - x1bar) * (x2i - x2bar)
# so it is a single summation, which means we can partition n into independent parts. 
# say we split n into 2 parts, [n1,n2]. 
# say n1 = n//2, n2 = n
# we first get the means of [n1,n2] and then get the full mean by taking the mean of means.
# then we can compute:
# sum__1_n1 (x1i - x1bar) * (x2i - x2bar)
# sum_n1_n2 (x1i - x1bar) * (x2i - x2bar)
# then add these and divide by n to get cov(x1, x2). 

# we need to do a first pass to get the means.
# x1bar = 1 / n sum_1_n x1i 
# we will probably have one batch with a weird size so we can rewrite:
# x1bar = 1 / n ( sum_1_n1 x1i + sum_n1_n2 x1i )
# x1bar = 1 / n ( (n1-0) * mean_x1(1,n1) + (n2-n1) * mean_x1(n1,n2))
# x1bar = (n1-0) / n * mean_x1(1,n1) + (n2-n1) / n * mean_x1(n1,n2)
# so we can get the means of each batch and remember their weights. 

# the second step is to go over the data AGAIN and calculate its std 
# var = 1/n sum_1_n (x1i - x1bar)^2
# std = SQRT var
# we can compute the var of each chunk.
# var = 1/n ( sum_1_n1 (x1i - x1bar)^2  +  sum_n1_n2 (x1i - x1bar)^2)
# mse(n1,n2,x1bar) = 1/(n2-n1) * sum_n1_n2 (x1i - x1bar)^2
# var = (n1-0)/n * mse(1,n1,x1bar) + (n2-n1)/n * mse(n1,n2,x1bar)

# the fact that we need 3 steps to go over the data can be seen by this function:
#def standardize(inp):
#    mu                  = np.mean(inp, axis = 0, keepdims = True)
#    std                 = np.std(inp - mu, axis = 0, keepdims = True)# calculate variance of features
#    x                   = (inp - mu) / std                             # scale the data
#    return x, mu, std
# mu depends on all x so that's one step. std depends on all x and mu, that's a second step.
# x depends on input, mu, and std so that is a third step. 



import numpy as np
import tkinter as tk
from tkinter import ttk
import zipfile, io, os, pickle
from PIL import Image, ImageTk


siz                 = 64     # picture length and width
SAVE_DIR            = r'C:\Users\i_hat\Desktop\bastl\py\deep_larn\anime_PCA'
IMG_DIR             = r'C:\Users\i_hat\Desktop\losable\anime_face_400\images'


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


def get_chunk_means(data_chunk): # get the mean of the chunk: 1/(n2-n1) sum_n1_n2 x1i
    b_xr, b_xg, b_xb = data_chunk
    return (np.mean(b_xr, axis = 0, keepdims = True), np.mean(b_xg, axis = 0, keepdims = True), np.mean(b_xb, axis = 0, keepdims = True))


def get_sample_weights(inds): # get the weights of the samples: ni / n  
    n = inds[-1][-1]
    return [(end - start) / n for start, end in inds]


def get_chunk_mse(data_chunk, mu_r, mu_g, mu_b):
    _xr, _xg, _xb = data_chunk
    mse_r = np.mean(np.square(_xr - mu_r), axis = 0, keepdims = True)
    mse_g = np.mean(np.square(_xg - mu_g), axis = 0, keepdims = True)
    mse_b = np.mean(np.square(_xb - mu_b), axis = 0, keepdims = True)
    return mse_r, mse_g, mse_b


def get_chunk_cov(data_chunk, mu_r, mu_g, mu_b, std_r, std_g, std_b, n):
    xr, xg, xb = data_chunk
    _xr = (xr - mu_r) / std_r # standardize the data using the mean and std
    _xg = (xg - mu_g) / std_g
    _xb = (xb - mu_b) / std_b
    return (_xr.T @ _xr / n, _xg.T @ _xg / n , _xb.T @ _xb / n)


def eig(S):
    va, vc          = np.linalg.eigh(S)  
    _sorted         = np.argsort(-va) # sorting them in decrasing order
    va              = va[_sorted]
    vc              = vc[:, _sorted]
    return (va, vc)


if False:

    # S T E P   1   :   G A T H E R   T H E   M E A N S
    chunk_means = []
    inds = generate_chunk_inds(63_565, 10_000)
    for start, end in inds:
        chunk_means.append(get_chunk_means(get_data_chunk(start, end)))

    # get the total mean
    w = get_sample_weights(inds)
    mu_r = np.zeros_like(chunk_means[0][0])
    mu_g = np.zeros_like(chunk_means[0][1])
    mu_b = np.zeros_like(chunk_means[0][2])
    for i in range(len(w)):
        mu_r += w[i] * chunk_means[i][0]
        mu_g += w[i] * chunk_means[i][1]
        mu_b += w[i] * chunk_means[i][2]


    # S T E P   2   :   G A T H E R   T H E   S T D S
    chunk_mses = []
    for start, end in inds:
        data_chunk = get_data_chunk(start, end)
        chunk_mses.append(get_chunk_mse(data_chunk, mu_r, mu_g, mu_b))

    var_r = np.zeros_like(chunk_mses[0][0])
    var_g = np.zeros_like(chunk_mses[0][1])
    var_b = np.zeros_like(chunk_mses[0][2])
    for i in range(len(chunk_mses)):
        var_r += w[i] * chunk_mses[i][0]
        var_g += w[i] * chunk_mses[i][1]
        var_b += w[i] * chunk_mses[i][2]
    std_r = np.sqrt(var_r)
    std_g = np.sqrt(var_g)
    std_b = np.sqrt(var_b)
    std_r[std_r == 0.0] = 1.0
    std_g[std_g == 0.0] = 1.0
    std_b[std_b == 0.0] = 1.0


    # S T E P   3   :   C O V A R I A N C E 
    chunk_covs = []
    for start, end in inds:
        data_chunk = get_data_chunk(start, end)
        chunk_covs.append(get_chunk_cov(data_chunk, mu_r, mu_g, mu_b, std_r, std_g, std_b, inds[-1][-1]))

    # add the chunk covs 
    cov_r = np.zeros_like(chunk_covs[0][0])
    cov_g = np.zeros_like(chunk_covs[0][1])
    cov_b = np.zeros_like(chunk_covs[0][2])
    for i in range(len(chunk_covs)):
        cov_r += chunk_covs[i][0]
        cov_g += chunk_covs[i][1]
        cov_b += chunk_covs[i][2]

    del chunk_covs, chunk_means, chunk_mses, inds, w, data_chunk, var_r, var_g, var_b, start, end

    # S T E P   4   :   P C A
    _, vecsr = eig(cov_r)
    _, vecsg = eig(cov_g)
    _, vecsb = eig(cov_b)

    pickle.dump([vecsr, vecsg, vecsb, mu_r, mu_g, mu_b, std_r, std_g, std_b], open(os.path.join(SAVE_DIR, 'anime_400_PCA_3.pkl'), 'wb'))


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


'''
- - - PCA notes and observations, in regards to the R/G/B separate version.
0. hair color
1. make hair short by subtracting ligthness from the bottom of the pic

2. rotate in one color channel
3. rotate in another color channel

4. cartooniness / head shape & eye size 
5. bangs

6. remove bug eyed doll + some tilt
7. more tilt (cancels out some of 6)

8. bug eyes 
9. cancel bug eyes + turn and elongate head

10 + 11 more head tilt

12 + 13 head wideness

14. head elevation + cancel smile
15. smile

16, 17, 18  head tilt + no smile (makign both neg gives big smile)

19, 20, 21, 22, 23: head shape 

24, 25: tilt + smile

26, 27, 28, 29: eye size + smile







@ @ @ each PCA eigenvector is a picture being added on the mean image.
- when the slider goes from positive to negative, we see the colors invert. 
    - this can be seen since the colors invert: green/purple changes to purple/green.
    - this can even be noted in the "rotation" features - meaning the magic is in adding an image.

- now, interestingly, we see perceptual "feature" changes. so far the notable features I saw have been:
    1. hair color: 1,2
    2. face rotation: 3,4
    3. face shape
    4. add smile
    but it seems that all these "features" are the results of adding a picture to the mean image. 

- for example, even the longer face feature is like an added chin, subtracting it gives a shadow where the chin was.
- i turned off adding the mean, now we can see what it's adding. 
    - crazy how a picture looks rotated by add/sub another pic.
    - would this still happen if we did PCA on the combined RGB vector?

- now check this out: rotating a face will mean it has to change the color of its eyes. 
    - which explains why the second feature is again rotation, but in diff colors.
    - specifically, features 3 and 4 have the eyes at a diagonal. it's like it subtracts some eye from the mean and adds more eye to the bottom to look rotated
    - subtracting this vector does the opposite, meaning adds eye to the opposite eye, making the face look rotated.
    - because of the color values used in the eye added, we can't have this vector explain the OTHER rotation for that color. 
    - that is why the feature right under it is also rotation, but for a different color! 

- this color-feature duality even explains why the first two features deal with hair: 
    - the first is long hair, it adds white to the entire background, as can be seen in the reconstruction without means. 
    - second is short hair which leaves half the background as-is.
- the first feature is definitely long hair lightness, pure and simple.
- the second feature is SHORT hair lightness: but to be orthogonal from the first feature, it has NEGATIVE weights in the bottom! 
    - this changes the bottom background color!
    - if you add this feature to the first feature, both features agree to add light on top, but f1 says bottom is light, f2 subtracts it away w neg weights.
    - in fact the weights of f2 are more heavy on the negative side. it adds very little lightness to the top hair. 
    - so f1 is the hair color, f2 is "subtract bottom hair". 

- some features seem to be "negative" features as in they are a lot stronger in the negative side, 
    - so they subtract lightness from the image (since they have positive eigenvalues)

- feature 4 does a lot: it magnifies a specific hair style, but also removes chin, forehead, and eye. 
    - this has the effect of giving the head a more cartoony vs real look. 

- i think feature 5 is bangs because it's stronger negatively and it subtracts from the light mean face to create darker bangs. 

- f6 seems to be creating a specific face position, and eye shape. but subtracting it leaves a bug-eyes doll. 
    - these features are interesting to interpret with and without the mean. 
    - with the mean we see that it cancels some of the bottom eye to make the eye appear higher up. 
    - because this feature is more interpretable in the negative direction, i would say it's the "remove bug eye doll look" feature. 

- f7 is rotation again. 

- f8 is bug eyes which it does by having large negative weights around the eyes, makign them black. 

- f9 seems to cancel the bug eyes while morphing other stuff like face shape and direction. 
    - putting f8 with negative f9 gives super bug eyes. 


HERE IS SOME INTUITION ON WHAT IS GOING ON WITH THE HAIR.

let's simplify, but without losing generality. Say we have an 8-bit picture of our anime character. 
now say we have the same mean picture. ok. 
when we add lightness to the hair, all the hair pixels become whiter. that is about half the picture. 
in our 8-bit image, that's like 8 pixels becoming white. 
now say we go super 8-bit and our new anime character image is just 4 pixels, and 3 of them are hair!
sine hair is correlated, they are all about the same value. so when the hair color becomes white, 
those 3 pixels are all near 255. when hair is black they are all near 0. 
now we can actually plot these points in a 3D grid. since hair color is usually the same or highly correlated,
the 3 pixels corresponding to hair color can be mapped on a 3D graph along the line z = y = x. 
meaning when hair is white, its color values are near 255 and the point is near (255,255,255). 
when hair is black, its color values are near (0,0,0). all of these points form a SINGE LINE IN 3D SPACE!!!
what this tells us is that hair actually forms a 1D subspace in our 4096D space. 
of course not all pixels of each image correspond to hair, so those pixels will stray from the 1D line. 
this is like saying imagine i have a second 4-pixel image of an anime character, but only 2 pixels are hair now. 
so two of its dimensions will be near the line, the other dimension will be elsewhere.
now because hair takes up half the picture, a lot of those points will be part of the hair manifold. 
and they are all correlated. 
so, roughly speaking (we are in stats after all), the variation in half of the image (since half the image is hair)
is explained by movement along a 1D manifold. give or take some pixels. 
why does this explain most of the variation? because the rest of the image is basically noise compared to how 
smoothly structured the hair is - there are eyes, mouths, shading, etc. 

we can take this intuition and extrapolate it. for example, the mouth, when present, can be seen as a red blob.
let's go back to the 8-bit image. the mouth can be seen as 1 pixel. some pictures have it, and others don't. 
those that have it will be clustered around a point in 1D corresponding to the red color. others will be elsewhere, 
probably wherever corresponds to the skin color. 

the mouth / smile approximation is even rougher than the hair one because mouths are in different shapes, sizes, 
and appear in different areas of the image. so their clustering will be nowhere near as structured as that of hair, 
yet it exists anyway in features with smaller eigenvalues. 

thus we can now understand that moving along the 1D hair manifold corresponds to increasing the pixels in all the 
hair dimensions - all the pixels corresponding to hair. 

feature 2 removes hair from the picture bottom. so imagine in our 4 pixel image, this would mean removing 1 pixel of hair.
on our 3D graph, say the x-axis corresponds to this pixel. then all the images with short hair will still be along 
the line z = y but their x axis is now free to display non-hair. 

up nex is head tilt. we know that head tilt is 1 image but there are like 8 features for it, and different colors of it. 
it's actually easy as pie when we think about it the same way with hair. tilted heads have one eye lower, so the part 
above the eye is going to be yellow in pixel space. 
so that bundle of pixels will be yellow whereas others will be eye-colored there.

now imagine you are an eigenvector. you start at zero and you point at images of tilted anime heads (tilted ot one side, 
because the ones tilted to the other side are in aa different direction). to get there you need to point in the yellow direction. 
this implies the 3 eigenvectors for r,g,b will point in whatever value Yellow is. if we subtract this value, it actually becomes
blue. so subtracting this eigenvector undoes its feature which comes off as applying the blue color where there was yellow, 
and we see that as the eye growing/moving in that direction which looks like rotating! but this happens in only a certain color channel,
the color channel of that character's eye. it does not account for the tilted heads of characters with different eye colors. 
that is why a second feature of tilted heads is needed - one that subtracts and adds a different color, allowing those characters 
to tilt their heads. 

i think at this stage we have a pretty good understanding of what is going on. 
when we have a ton of images, if they share similarities, those similarities come off as bundles of pixels with similar values. 
those bundles for clusters in space. a cluster of pixels means we don't have a wild distribution of points from 0-255 in 
the dimensions of that cluster! (or more accurately, a manifold rather than a cluster)
and because that cluster is there, if it is distinct enough from the mean image, meaning has a high pixel value difference, 
then we can get a large eigenvalue from the covariance matrix pointing at that cluster, and that will exactly be the eigenvector!

and adding the eigenvector will produce that feature. subtracting it will remove the feature (which might be unrecognizeable)
think back on the 4-bit pixel - the mouth is one red pixel, and all images who have a mouth form a cluster in the mouth area 
while the rest are in the yellow no-mouth area. then the variance matrix will have an eigenvector pointing at the cluster of points 
with red mouths. this analogy extends to real images where a mouth is just something like 100 pixels of near-red. 

all this talking is based on the assumption that the manifold is linear, like hair. 
the most likely alternative is just a normally distributed point cloud. 

NOW 

performing PCA does this: since we see that the correlation between the hair pixels is about 1, meaning if one changes then the 
others change the same, and they are on a 1D manifold at z=y=x, the eigenvector will point in that direction. so going along
the eigenvector will change all the pixels the same, and not change any other pixels. so in terms of the 4 pixel image, this
eigenvector's values will be [1,1,1,0]. 
similarly, f2's eigenvector will be something like [0.5,0.5,-1,0]. 

features 2,3 for face tilting will each have values that remove the upper part of the eye (dark values) and replace it with 
skin color values (yellow)

similarly other features will have values that correspond to adding that feature on to the image. like a smile will 
have high values near the mouth area because in picture space there is a cluster there where a lot of point have smiles. 
imagine our 4 pixel image set. pictures with smiles will all have a dark value in the smile pixel, others will have yellow,
making them two clusters located apart from each other on the axis corresponding to the smile pixel. 

takeaway:
eigenvectors draw features by adding values to pixels' rgb channels. 
the pixels are found by pointing at clusters of points from the origin.
clusters are created when a lot of images have the same color in the same place (mouth, eyes)
-which means the cluster has a very narrow rgb range in those pixels' dimensions.
features come in pairs because one will draw it, the second will undraw it while adding another feature.

so a lot of these features depend on manipulation of color channels, which causes features to be spread amongst consecutive eigenvectors.
how did the 3 color channels coordinate to create these featuers when we did PCA on r,g,b channels separately?
well that is because the 3 channels are perceptually identical. we can view the rgb channels as BW images and they are near identical,
except for their actual values. 
so in image space, the separate r, g, b, manifolds would have similar shape. thus all 3 eigenvectors agree hair is the main factor of variation.
but perhaps they would disagree on other features, and that is why we get mixtures of face turning and smiles? 
this question can be answered by investigating PCA on BW images, as well as PCA on the combined RGB channels. 


ok i did PCA on the combined RGB channels. 
the results are much cleaner but a bit less interpretable.. i think?
some features are clear to see like lighting, eye changes, hair changes. but the features are very intercoupled. 
this might be because the primary feature explains a lot of the variation, so the rest of the residual variation expects to be adjusted by the main vector.
for example, f1 is still hair lightness. 
but f2 is hair color, hair size, and eye size. 
a lot of features change head shape, eye shape, and colors. so it's a bit hard to interpret them. 

there are 2 possibilities:
1. when an eigenvector points in the direction of short hair, it also captures some hair color with it. 
2. the size of variation explained by hair shortness and hair orangeness are similar so the eigenvector points in between them. 
    thus changing this eigenvector brings forth both features. this seems more reasonable 
    ...
    why don't we get this behavior in the separate rgb case? because we do PCA on the channels separately so color is basically thrown away
    and we only deal with intensity. and features almost never appear as a single R, G, or B channel. 
    this actually makes a lot of sense. 
    in the combined rgb channel we deal with 'color' being a feature a lot so a lot of the vectors deal with color transforms, and some shape features. 
    but once again, in the separate rgb case we do not have colors but rather intensity, which is shared across all 3 channels. 
    i think this now begs the question: what if we do the analysis on BW images? the features should be similar to the separate rgb case. 
    but not exactly since we won't have pairs of vectors undoing each other's color contributions. 


Ok just did the analysis for BW. the features are identical to the separate rgb ones. 
they are easier to interpret because we aren't distracted by colors, and instead only see shadows/lights and features. 

takeaway
- BW is better for figuring out what the features are. its features are the same as doing PCA on r,g,b separately. 


I had a question before:
- check which pixels in the image have highest variance. also can think of a way to check covariance, for a given pixel

the answer is given to us implicitly in the eigenvectors. consider the eigenvector for hair, and all the pixels that light up
when its value is high. those all have positive weights, the rest of the weights are zero. this tells us that all those pixels
have similar covariance 


After losing everything on my computer after downloading a GNU tool, I come back to start fresh from this project. 

By mistake I managed to somehow plot the movements of 3 separate channels' eigenvectors. sometimes 2 moved together and 1 in the opposite direction!

although they catch similar features, seems the eigenvalues may be opposite. then their combo will only move 1/3 as much, as they'll cancel out.
- now that i think about it, this shows us the 3 eigenvectors are similar but nowhere the same. heads tilt different amounts. so concatenated, 
these will be very bad. in fact i think thats why we get the trippy colors! because the color channels disagree so we always get 1 channel going in the other direction.
- ok i fixed the problem by flipping the sign of the eigenvector if they are opposite, thus aligning each triple. 
- some features still differ by different amounts. that causes discoloration. 

Not much left to do at this point - we can either view each color channel's eigenvector separately, or BW, or assume BIG DATA will average out the differences.
Actually the only real way to get rid of this is to do PCA on the combined 12k dimensional vectors. 

Also you can easily tell that one eigenvector has gone berserk when the picture becomes too trippy in color. 

Either way, it is a heckin lot better now. 

I guess the true way to look at features is using the 12k vectors. but those get distracted by colors. but color is a feature. 
so what is the best representation? i think BW for features, and 12k for features with colors in mind. 

after all that, i think the first candidate for a real anime face creator is the 12k PCA. 

after even more thinking i think the 12k version is the only serious one. BW is too simple, colorful is too trippy. 12k is actually
somewhat practical and looks good. for PCA_2,3 we can't really average the eigenvectors as that would just align all the changes 
to be the same for each color channel, which kinda defeats the pruposes of there being 3 of them. So 12k, having tamed color channels,
is the only usable one. 


'''
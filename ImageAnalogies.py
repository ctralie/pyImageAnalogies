"""
Programmer: Chris Tralie
Purpose: To implement the basic features of Image Analogies[1] in Python,
using exact nearest neighbors (pyflann would be faster, but has trouble with Python3)

Image Analogies
A. Hertzmann, C. Jacobs, N. Oliver, B. Curless, D. Salesin.
SIGGRAPH 2001 Conference Proceedings.
"""
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import imageio
import argparse
from skimage.transform import pyramid_gaussian
from tqdm import tqdm
import time

imresize = lambda x, shape: skimage.transform.resize(x, shape, anti_aliasing=True, mode='constant')

def rgb2gray(rgb):
    """
    Convert an RGB image to grayscale
    Parameters
    ----------
    rgb: ndarray(M, N, 3)
        An M x N color image
    Returns
    -------
    gray: ndarray(M, N)
        The corresponding grayscale image
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def read_image(filename):
    """
    Load an image from disk
    Parameters
    ----------
    filename: string
        Location of image
    Returns
    -------
    I: ndarray(M, N, 3, dtype=float)
        A color channel image converted to floating
        point [0, 1] per channel
    """
    I = imageio.imread(filename)
    I = np.array(I, dtype=np.float32)/255.0
    if len(I.shape) == 2:
        # Stack grayscale to fake a color image
        I = I[:, :, None]
        I = np.concatenate((I, I, I), axis=2)
    if I.shape[2] > 3:
        # Only use RGB
        I = I[:, :, 0:3]
    return I

def write_image(I, filename):
    """
    Write a floating point image to disk after rounding
    to the nearest byte for each color channel
    Parameters
    ----------
    I: ndarray(M, N, 3, dtype=float)
        MxN color image in floating point [0, 1] per channel
    filename: string
        The path to which to save the file
    """
    IRet = I*255.0
    IRet[IRet > 255] = 255
    IRet[IRet < 0] = 0
    IRet = np.array(IRet, dtype=np.uint8)
    imageio.imwrite(filename, IRet)

def get_patches(I, dim, i, j):
    """
    Sample patches from a (possibly color) image

    Parameters
    ----------
    I: ndarray(M, N, ...)
        Image from which to sample patches
    dim: int
        Dimension of patches
    i: ndarray(n_patches, dtype=int)
        Row of patch
    j: ndarray(n_patches, dtype=int)
        Column of patch
    
    Returns
    -------
    patches: ndarray(n_patches, dim*dim)
        Array of patches
    """
    n_patches = i.size
    pix = np.arange(dim)
    ii, jj = np.meshgrid(pix, pix, indexing='ij')
    ii = ii.flatten()
    jj = jj.flatten()
    ii = (i[:, None] + ii[None, :]).flatten()
    jj = (j[:, None] + jj[None, :]).flatten()
    patches = I[ii, jj, ...]
    nch = 1
    if len(I.shape) > 2:
        nch = I.shape[2]
    return np.reshape(patches, (n_patches, dim*dim*nch))


def get_causal_patches(I, dim, i, j):
    """
    Assuming dim is odd, return L-shaped patches that would
    occur in raster order

    Parameters
    ----------
    I: ndarray(M, N)
        Single channel image
    dim: int
        Dimension of patches
    i: ndarray(n_patches, dtype=int)
        Row of patch
    j: ndarray(n_patches, dtype=int)
        Column of patch
    """
    P = get_patches(I, dim, i, j)
    carea = (dim*dim)//2 # Causal area
    P = P[:, 0:carea]
    return P

def get_coherence_match(x0, BL, BpL, B2, Bp2, BpLidx, dim, i, j):
    """
    Return the best coherence match based on pixels that have already been chosen

    Parameters
    ----------
    x0: ndarray(DimFeatures)
        The feature vector of the pixel that's being filled in
    BL: ndarray(M, N)
        B image at this level
    BpL: ndarray(M, N)
        B' image at this level
    B2: ndarray(M, N)
        B image one level up, resized to this resolution (or empty array if there is nothing above)
    Bp2: ndarray(M, N)
        B' image one level up, resized to this resolution (or empty array if there is nothing above)
    BpLidx: ndarray(MxN)
        An MxN array of raveled indices from which pixels have
        been drawn so far at this level
    dim: int
        Dimension of patch
    i: int
        Row of pixel currently being processed
    j: int
        Column of pixel currently being processed
    Returns
    -------
    idxMin: int
        Index into the array of patches of the closest coherent patch
    minDistSqr: float   
        Squared distance of the closest patch
    """
    area  = dim*dim # Area
    carea = area//2 # Causal area
    M = BL.shape[0]
    N = BL.shape[1]
    minDistSqr = np.inf
    idxmin = [-1, -1]
    d = (dim-1)//2
    [dI, dJ] = np.meshgrid(np.arange(dim), np.arange(dim), indexing='ij')
    dI = np.array(dI.flatten()[0:carea], dtype = np.int64) - d
    dJ = np.array(dJ.flatten()[0:carea], dtype = np.int64) - d
    #TODO: Vectorize code below
    sz = area + carea  #B, Bp
    if B2.size > 0:
        sz += area*2
    x = np.zeros(sz)
    for n in range(dI.size):
        #Indices of pixel picked for neighbor
        ni = BpLidx[int(dI[n]+i), int(dJ[n]+j)][0]
        nj = BpLidx[int(dI[n]+i), int(dJ[n]+j)][1]
        if ni == -1 or nj == -1:
            continue
        ni = int(ni - dI[n])
        nj = int(nj - dJ[n])
        if ni - dim < 0 or nj - dim < 0 or ni + dim >= M or nj + dim >= N:
            continue

        #Full patch B
        x[0:area] = BL[ni-d:ni+d+1, nj-d:nj+d+1].flatten()
        #Causal patch B'
        x[area:area+carea] = BpL[ni-d:ni+d+1, nj-d:nj+d+1].flatten()[0:carea]
        if B2.size > 0:
            #Use multiresolution features
            x[area+carea: area*2+carea] = B2[ni-d:ni+d+1, nj-d:nj+d+1].flatten()
            x[area*2+carea:] = Bp2[ni-d:ni+d+1, nj-d:nj+d+1].flatten()

        distSqr = np.sum((x - x0)**2)
        if distSqr < minDistSqr:
            minDistSqr = distSqr
            idxmin = [ni, nj]
    return (idxmin, minDistSqr)


def imanalogy(A, Ap, B, Kappa = 0.0, NLevels = 3, KCoarse = 5, KFine = 5, n_jobs = None, debug_images=False, use_ann=True):
    """
    Perform image analogies
    Parameters
    ----------
    A: ndarray(M, N, ...)
        A floating point image to use as A, with an optional channel dimension
    Ap: ndarray(M, N, 3)
        A floating point image to use as A', with an optional channel dimension
    B: ndarray(M, N, 3)
        A floating point image to use as B, with an optional channel dimension
    Kappa: float
        The coherence factor (how much to promote nearby patches
        over good matches)
    NLevels: int
        The number of levels to use in a multiresolution reconstruction
    KCoarse: int
        Dimension of square patches to sample at the coarsest level
        of the multiresolution pyramid
    KFine: int
        Dimension of square patches to sample at the finest level
        of the multiresolution pyramid
    n_jobs: int
        Number of parallel processes to run for nearest neighbor search (deafult None)
    use_ann: bool
        If True, use approximate nearest neighbors.  If False, use exact nearest neighbors
    
    Returns
    -------
    Bp: ndarray(M, N, ...)
        A floating point image of the synthesized image analogy, with the corresponding
        channel dimension
    """
    #Make image pyramids
    channel_axis = {}
    if len(A.shape) > 2:
        channel_axis = {"channel_axis":-1} # pyramic_gaussian needs this parameter for color images
    AL = tuple(pyramid_gaussian(A, NLevels, downscale = 2, **channel_axis))
    ApL = tuple(pyramid_gaussian(Ap, NLevels, downscale = 2, **channel_axis))
    BL = tuple(pyramid_gaussian(B, NLevels, downscale = 2, **channel_axis))
    ApLColor = [img for img in ApL]
    if len(A.shape) > 2: # For color images, use grayscale features, as recommended in the paper
        AL = [rgb2gray(img) for img in AL]
        ApL = [rgb2gray(img) for img in ApL]
        BL = [rgb2gray(img) for img in BL]
    BpL = [] # B' values in pyramid (grayscale)
    BpLColor = [] # B' values in pyramid (color)
    BpLidx = [] # Indices of nearest neighbors at each pixel in pyramid
    print("BL:")
    for i in range(len(BL)):
        print(BL[i].shape)
        BpL.append(np.zeros(BL[i].shape))
        shape = list(BL[i].shape)
        if len(A.shape) > 2:
            shape.append(A.shape[2])
        BpLColor.append(np.zeros(shape))
        BpLidx.append(-1*np.ones((BL[i].shape[0], BL[i].shape[1], 2), dtype=int))
    print("AL:")
    for i in range(len(AL)):
        print(AL[i].shape)

    #Do multiresolution synthesis
    for level in range(NLevels, -1, -1):
        print("Doing level", level)
        total_time = 0
        ## Step 0: Work out dimension of patches
        KSpatial = KFine
        if level == NLevels:
            KSpatial = KCoarse
        d = (KSpatial-1)//2
        area  = KSpatial*KSpatial # Area
        carea = area//2 # Causal area
        sz = area + carea  #B, Bp
        if level < NLevels:
            sz += area*2

        ## Step 1: Make features
        ## Step 1a: Determine location of patches
        shape = [AL[level].shape[0]-KSpatial+1, AL[level].shape[1]-KSpatial+1]
        ipatch, jpatch = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        ipatch = ipatch.flatten()
        jpatch = jpatch.flatten()
        ## Step 1b: Get patches
        APatches = get_patches(AL[level], KSpatial, ipatch, jpatch)
        ApPatches = get_causal_patches(ApL[level], KSpatial, ipatch, jpatch)
        X = np.concatenate((APatches, ApPatches), 1)
        B2 = np.array([])
        Bp2 = np.array([])
        if level < NLevels:
            # Step 1c: Use multiresolution features if necessary
            A2 = imresize(AL[level+1], AL[level].shape)
            Ap2 = imresize(ApL[level+1], ApL[level].shape)
            A2Patches = get_patches(A2, KSpatial, ipatch, jpatch)
            Ap2Patches = get_patches(Ap2, KSpatial, ipatch, jpatch)
            X = np.concatenate((X, A2Patches, Ap2Patches), 1)
            B2 = imresize(BL[level+1], BL[level].shape)
            Bp2 = imresize(BpL[level+1], BpL[level].shape)
        # Step 1d: Setup nearest neighbor structures
        nn = None
        if use_ann:
            from pynndescent import NNDescent
            nn = NNDescent(X)
        else:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=n_jobs).fit(X)
        

        ## Step 2: Fill in the first few scanLines to prevent the image
        ## from getting crap in the beginning
        if level == NLevels:
            I = np.array(ApL[level]*255, dtype = np.uint8)
            I = imresize(I, BpL[level].shape)
            BpL[level] = np.array(I/255.0, dtype = np.float64)
        else:
            I = np.array(BpL[level+1]*255, dtype = np.uint8)
            I = imresize(I, BpL[level].shape)
            BpL[level] = np.array(I/255.0, dtype = np.float64)

        ## Step 3: Fill in the pixels in scanline order
        F = np.zeros(sz)
        for i in tqdm(range(d, BpL[level].shape[0]-d)):
            for j in range(d, BpL[level].shape[1]-d):
                #Make the feature at this pixel
                #Full patch B
                F[0:area] = BL[level][i-d:i+d+1, j-d:j+d+1].flatten()
                #Causal patch B'
                F[area:area+carea] = BpL[level][i-d:i+d+1, j-d:j+d+1].flatten()[0:carea]
                if level < NLevels:
                    #Use multiresolution features
                    F[area+carea: area*2+carea] = B2[i-d:i+d+1, j-d:j+d+1].flatten()
                    F[area*2+carea:] = Bp2[i-d:i+d+1, j-d:j+d+1].flatten()
                #Find index of most closely matching feature point in A
                tic = time.time()
                if use_ann:
                    idx, dist = nn.query(F[None, :], k=1)
                else:
                    dist, idx = nn.kneighbors(F[None, :])
                total_time += time.time()-tic
                distSqr = dist**2
                idx = int(idx[0][0])
                idx = [ipatch[idx], jpatch[idx]]
                """
                if Kappa > 0:
                    #Compare with coherent pixel
                    (idxc, distSqrc) = get_coherence_match(F, BL[level], BpL[level], B2, Bp2, BpLidx[level], KSpatial, i, j) 
                    fac = 1 + Kappa*(2.0**(level - NLevels))
                    if distSqrc < distSqr*fac*fac:
                        idx = idxc
                """
                BpLidx[level][i, j, :] = idx
                BpL[level][i, j] = ApL[level][idx[0]+d, idx[1]+d]
                BpLColor[level][i, j, ...] = ApLColor[level][idx[0]+d, idx[1]+d, :]
            if i%20 == 0 and debug_images:
                write_image(BpLColor[level], "%i.png"%level)
        print("Time nearest neighbors:", total_time)
        if debug_images:
            plt.subplot(122)
            plt.imshow(BpLidx[level][:, :, 0], cmap = 'Spectral')
            plt.title("Y")
            plt.subplot(121)
            plt.imshow(BpLidx[level][:, :, 1], cmap = 'Spectral')
            plt.title("X")
            plt.savefig("%i_idx.png"%level, bbox_inches = 'tight')
    return BpLColor[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--A', type=str, required=True, help="Path to A image")
    parser.add_argument('--Ap', type=str, required=True, help="Path to A' Image")
    parser.add_argument('--B', type=str, required=True, help="Path to B Image")
    parser.add_argument('--Bp', type=str, required=True, help="Path to which to save B' Image")
    parser.add_argument('--Kappa', type=float, default=0.0, help="Coherence parameter (higher means coherence is favored over matches)")
    parser.add_argument('--NLevels', type=int, default=2, help="Number of levels to use in multiresolution pyramid")
    parser.add_argument('--KCoarse', type=int, default=5, help="Resolution of coarse patches")
    parser.add_argument('--KFine', type=int, default=5, help="Resolution of finer patches")
    parser.add_argument('--njobs', type=int, default=1, help="Number of parallel processes to use in nearest neighbor search")
    parser.add_argument('--debugImages', type=int, default=1, help="Whether to output all images in pyramid and chosen indices progressively as B' is being constructed")
    parser.add_argument('--ann', type=int, default=1, help="If 1, use approximate nearest neighbors (requires pynndescent).  If 0, revert to sklearn exact nearest neighbors")
    opt = parser.parse_args()

    A = read_image(opt.A)
    Ap = read_image(opt.Ap)
    B = read_image(opt.B)
    Bp = imanalogy(A, Ap, B, Kappa=opt.Kappa, NLevels=opt.NLevels, KCoarse=opt.KCoarse, KFine=opt.KFine, n_jobs=opt.njobs, debug_images=bool(opt.debugImages), use_ann=(opt.ann == 1))
    write_image(Bp, opt.Bp)
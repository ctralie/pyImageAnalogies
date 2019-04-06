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
import scipy.io as sio
import scipy.misc
import skimage.transform
import imageio
import argparse
from sklearn.neighbors import NearestNeighbors
from skimage.transform import pyramid_gaussian

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

def readImage(filename):
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
    return I

def writeImage(I, filename):
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

def getPatches(I, dim):
    """
    Given an an MxN single channel image I, get all dimxdim dimensional
    patches
    Parameters
    ----------
    I: ndarray(M, N)
        Single channel image
    dim: int
        Dimension of patches
    Returns
    -------
    P: ndarray((M-dim+1), (N-dim+1), dimxdim))
        Array of flattened patches
    """
    #http://stackoverflow.com/questions/13682604/slicing-a-numpy-image-array-into-blocks
    shape = np.array(I.shape*2)
    strides = np.array(I.strides*2)
    W = np.asarray(dim)
    shape[I.ndim:] = W
    shape[:I.ndim] -= W - 1
    if np.any(shape < 1):
        raise ValueError('Window size %i is too large for image'%dim)
    P = np.lib.stride_tricks.as_strided(I, shape=shape, strides=strides)
    P = np.reshape(P, [P.shape[0], P.shape[1], dim*dim])
    return P

def getCausalPatches(I, dim):
    """
    Assuming dim is odd, return L-shaped patches that would
    occur in raster order
    Parameters
    ----------
    I: ndarray(M, N)
        Single channel image
    dim: int
        Dimension of patches
    """
    P = getPatches(I, dim)
    k = int((dim*dim-1)/2)
    P = P[:, :, 0:k]
    return P

def getCoherenceMatch(X, x0, BpLidx, dim, i, j):
    """
    Return the best coherence match
    Parameters
    ----------
    X: ndarray(M, N, DimFeatures)
        Array of feature vectors at each pixel
    x0: ndarray(DimFeatures)
        The feature vector of the pixel that's being filled in
    BpLidx: ndarray(MxN)
        An MxN array of raveled indices from which pixels have
        been drawn so far
    dim: int
        Dimension of patch
    i: int
        Row of pixel
    j: int
        Column of pixel
    Returns
    -------
    idxMin: int
        Index into the array of patches of the closest coherent patch
    minDistSqr: float   
        Squared distance of the closest patch
    """
    k = int((dim*dim-1)/2)
    M = X.shape[0]
    N = X.shape[1]
    minDistSqr = np.inf
    idxmin = [-1, -1]
    [dJ, dI] = np.meshgrid(np.arange(dim), np.arange(dim))
    dI = np.array(dI.flatten()[0:k], dtype = np.int64) - (dim-1)/2
    dJ = np.array(dJ.flatten()[0:k], dtype = np.int64) - (dim-1)/2
    #TODO: Vectorize code below
    for n in range(dI.size):
        #Indices of pixel picked for neighbor
        ni = BpLidx[int(dI[n]+i), int(dJ[n]+j)][0]
        nj = BpLidx[int(dI[n]+i), int(dJ[n]+j)][1]
        if ni == -1 or nj == -1:
            continue
        ni = int(ni - dI[n])
        nj = int(nj - dJ[n])
        if ni < 0 or nj < 0 or ni >= M or nj >= N:
            continue
        x = X[ni, nj, :]
        distSqr = np.sum((x - x0)**2)
        if distSqr < minDistSqr:
            minDistSqr = distSqr
            idxmin = [ni, nj]
    return (idxmin, minDistSqr)


def doImageAnalogies(A, Ap, B, Kappa = 0.0, NLevels = 3, KCoarse = 5, KFine = 5, n_jobs = None, debugImages=False):
    """
    Perform image analogies
    Parameters
    ----------
    A: ndarray(M, N, 3)
        A floating point 3-channel image to use as A
    Ap: ndarray(M, N, 3)
        A floating point 3-channel image to use as A'
    B: ndarray(M, N, 3)
        A floating point 3-channel image to use as B
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
    
    Returns
    -------
    Bp: ndarray(M, N, 3)
        A floating point 3-channel image of the synthesized image analogy
    """
    #Make image pyramids
    AL = tuple(pyramid_gaussian(A, NLevels, downscale = 2))
    ApL = tuple(pyramid_gaussian(Ap, NLevels, downscale = 2))
    BL = tuple(pyramid_gaussian(B, NLevels, downscale = 2))
    BpL = [] # B' values in pyramid
    BpLidx = [] # Indices of nearest neighbors at each pixel in pyramid
    print("BL:")
    for i in range(len(BL)):
        print(BL[i].shape)
        BpL.append(np.zeros(BL[i].shape))
        BpLidx.append(-1*np.ones((BL[i].shape[0], BL[i].shape[1], 2), dtype=int))
    print("AL:")
    for i in range(len(AL)):
        print(AL[i].shape)

    #Do multiresolution synthesis
    for level in range(NLevels, -1, -1):
        KSpatial = KFine
        if level == NLevels:
            KSpatial = KCoarse
        #Step 1: Make features
        APatches = getPatches(rgb2gray(AL[level]), KSpatial)
        ApPatches = getCausalPatches(rgb2gray(ApL[level]), KSpatial)
        X = np.concatenate((APatches, ApPatches), 2)
        B2 = None
        Bp2 = None
        if level < NLevels:
            #Use multiresolution features
            A2 = imresize(AL[level+1], AL[level].shape)
            Ap2 = imresize(ApL[level+1], ApL[level].shape)
            A2Patches = getPatches(rgb2gray(A2), KSpatial)
            Ap2Patches = getPatches(rgb2gray(Ap2), KSpatial)
            X = np.concatenate((X, A2Patches, Ap2Patches), 2)
            B2 = imresize(BL[level+1], BL[level].shape)
            Bp2 = imresize(BpL[level+1], BpL[level].shape)
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=n_jobs).fit(np.reshape(X, [X.shape[0]*X.shape[1], X.shape[2]]))

        #Step 2: Fill in the first few scanLines to prevent the image
        #from getting crap in the beginning
        if level == NLevels:
            I = np.array(ApL[level]*255, dtype = np.uint8)
            I = imresize(I, BpL[level].shape)
            BpL[level] = np.array(I/255.0, dtype = np.float64)
        else:
            I = np.array(BpL[level+1]*255, dtype = np.uint8)
            I = imresize(I, BpL[level].shape)
            BpL[level] = np.array(I/255.0, dtype = np.float64)

        #Step 3: Fill in the pixels in scanline order
        d = int((KSpatial-1)/2)
        for i in range(d, BpL[level].shape[0]-d):
            print(i)
            for j in range(d, BpL[level].shape[1]-d):
                #Make the feature at this pixel
                #Full patch B
                BPatch = rgb2gray(BL[level][i-d:i+d+1, j-d:j+d+1, :])
                #Causal patch B'
                BpPatch = rgb2gray(BpL[level][i-d:i+d+1, j-d:j+d+1, :]).flatten()
                BpPatch = BpPatch[0:int((KSpatial*KSpatial-1)/2)]
                F = np.concatenate((BPatch.flatten(), BpPatch.flatten()))

                if level < NLevels:
                    #Use multiresolution features
                    BPatch = rgb2gray(B2[i-d:i+d+1, j-d:j+d+1, :])
                    BpPatch = rgb2gray(Bp2[i-d:i+d+1, j-d:j+d+1, :])
                    F = np.concatenate((F, BPatch.flatten(), BpPatch.flatten()))
                #Find index of most closely matching feature point in A
                dist, idx = nn.kneighbors(F[None, :])
                idx = int(idx[0])
                distSqr = dist**2
                idx = np.unravel_index(idx, (X.shape[0], X.shape[1]))
                if Kappa > 0:
                #Compare with coherent pixel
                    (idxc, distSqrc) = getCoherenceMatch(X, F, BpLidx[level], KSpatial, i, j)
                    fac = 1 + Kappa*(2.0**(level - NLevels))
                    if distSqrc < distSqr*fac*fac:
                        idx = idxc
                BpLidx[level][i, j, :] = idx
                BpL[level][i, j, :] = ApL[level][idx[0]+d, idx[1]+d, :]
            if i%20 == 0 and debugImages:
                writeImage(BpL[level], "%i.png"%level)
        if debugImages:
            plt.subplot(122)
            plt.imshow(BpLidx[level][:, :, 0], cmap = 'Spectral')
            plt.title("Y")
            plt.subplot(121)
            plt.imshow(BpLidx[level][:, :, 1], cmap = 'Spectral')
            plt.title("X")
            plt.savefig("%i_idx.png"%level, bbox_inches = 'tight')
    return BpL[0]

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
    opt = parser.parse_args()

    A = readImage(opt.A)
    Ap = readImage(opt.Ap)
    B = readImage(opt.B)
    Bp = doImageAnalogies(A, Ap, B, Kappa=opt.Kappa, NLevels=opt.NLevels, KCoarse=opt.KCoarse, KFine=opt.KFine, n_jobs=opt.njobs, debugImages = bool(opt.debugImages))
    writeImage(Bp, opt.Bp)
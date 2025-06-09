"""
Programmer: Chris Tralie
Purpose: To implement the basic features of Image Analogies[1] in Python

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

def get_patches(img, dim, i, j):
    """
    Sample patches from a (possibly color) image

    Parameters
    ----------
    img: ndarray(M + 2*pad, N + 2*pad, ...)
        Image from which to sample patches.  Assumed to already be padded
        by pad = dim//2 on both sides
    dim: int
        Dimension of patches, assumed to be odd
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
    patches = img[ii, jj, ...]
    nch = 1
    if len(img.shape) > 2:
        nch = img.shape[2]
    return np.reshape(patches, (n_patches, dim*dim*nch))


def get_causal_patches(img, dim, i, j):
    """
    Assuming dim is odd, return L-shaped patches that would
    occur in raster order

    Parameters
    ----------
    img: ndarray(M + 2*pad, N + 2*pad, ...)
        Image from which to sample patches.  Assumed to already be padded
        by pad = dim//2 on both sides
    dim: int
        Dimension of patches, assumed to be odd
    i: ndarray(n_patches, dtype=int)
        Row of patch
    j: ndarray(n_patches, dtype=int)
        Column of patch
    """
    P = get_patches(img, dim, i, j)
    carea = (dim*dim)//2 # Causal area
    if len(img.shape) > 2:
        carea *= img.shape[2]
    P = P[:, 0:carea]
    return P


def get_coherence_match(F, A1, Ap1, A2, Ap2, BpLidx, dim, i, j):
    """
    Return the best coherence match based on pixels that have already been chosen

    Parameters
    ----------
    F: ndarray(DimFeatures)
        The feature vector of the pixel that's being filled in
    A1: ndarray(M+2*pad, N+2*pad)
        A image at this level, padded
    Ap1: ndarray(M+2*pad, N+2*pad)
        A' image at this level, padded
    A2: ndarray(M+2*pad, N+2*pad)
        A image one level up, resized to this resolution (or empty array if there is nothing above)
        and padded
    Ap2: ndarray(M+2*pad, N+2*pad)
        A' image one level up, resized to this resolution (or empty array if there is nothing above)
        and padded
    BpLidx: ndarray(M, N, 2)
        Indices from which pixels have been drawn so far at this level
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
    M = BpLidx.shape[0]
    N = BpLidx.shape[1]
    area  = dim*dim # Area
    carea = area//2 # Causal area

    ## Step 1: Find valid offset patches in A'
    d = (dim-1)//2
    pixi = np.arange(max(i-d, 0), min(i-d+dim, M))
    pixj = np.arange(max(j-d, 0), min(j-d+dim, N))
    [i2, j2] = np.meshgrid(pixi, pixj, indexing='ij')
    i2 = i2.flatten()
    j2 = j2.flatten()
    # Compute indices to check
    idxs = BpLidx[i2, j2]
    # Make sure the indices were actually filled in
    i2 = i2[idxs[:, 0] > -1]
    j2 = j2[idxs[:, 0] > -1]
    idxs = idxs[idxs[:, 0] > -1, :]
    # Accounting for offset of the neighbor
    i2 = idxs[:, 0] - i2
    j2 = idxs[:, 1] - j2
    # Make sure these indices are within the bounds of A'
    M = Ap1.shape[0] - 2*d
    N = Ap1.shape[1] - 2*d
    valid = (i2 >= 0)*(i2 < M)*(j2 >= 0)*(j2 < N)
    i2 = i2[valid]
    j2 = j2[valid]

    if i2.size == 0:
        # No valid pixels!
        return [-1, -1], np.inf

    pix = np.arange(dim)
    ii, jj = np.meshgrid(pix, pix, indexing='ij')
    ii = ii.flatten()
    jj = jj.flatten()
    ii = (i2[:, None] + ii[None, :])
    shape = ii.shape
    jj = (j2[:, None] + jj[None, :])
    ii = ii.flatten()
    jj = jj.flatten()

    ## Step 2: Compute distances
    distsSqr = np.zeros(i2.size)
    Y = np.reshape(A1[ii, jj], shape)
    X = F[0:area]
    distsSqr += np.sum(X**2) + np.sum(Y**2, axis=1) - 2*(Y.dot(X)).flatten()

    Y = np.reshape(Ap1[ii, jj], shape)[:, 0:carea]
    X = F[area:area+carea]
    distsSqr += np.sum(X**2) + np.sum(Y**2, axis=1) - 2*(Y.dot(X)).flatten()

    if A2.size > 0:
        Y = np.reshape(A2[ii, jj], shape)
        X = F[area+carea:area*2+carea]
        distsSqr += np.sum(X**2) + np.sum(Y**2, axis=1) - 2*(Y.dot(X)).flatten()

        Y = np.reshape(Ap2[ii, jj], shape)
        X = F[area*2+carea:]
        distsSqr += np.sum(X**2) + np.sum(Y**2, axis=1) - 2*(Y.dot(X)).flatten()

    ## Step 3: Find and return the min distance
    idx = np.argmin(distsSqr)
    return [i2[idx], j2[idx]], distsSqr[idx]



def imanalogy(A, Ap, B, Kappa = 0.0, NLevels = 3, KCoarse = 5, KFine = 5, n_jobs = None, debug_images=False, use_ann=True, verbose=False):
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
    verbose: bool
        If True, print out debugging information
    Returns
    -------
    Bp: ndarray(M, N, ...)
        A floating point image of the synthesized image analogy, with the corresponding
        channel dimension
    """
    # Figure out nearest neighbor methods
    NNClass = None
    nn_query = None
    if use_ann:
        succeeded = True
        try:
            from pynndescent import NNDescent
        except:
            succeeded = False
            print("Warning: Could not import pynndescent: falling back to sklearn exact nearest neighbors")
        if succeeded:
            NNClass = lambda X: NNDescent(X)
            nn_query = lambda nn, F: nn.query(F[None, :], k=1)
    if not use_ann or not NNClass:
        from sklearn.neighbors import NearestNeighbors
        NNClass = lambda X: NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=n_jobs).fit(X)
        nn_query = lambda nn, F: list(nn.kneighbors(F[None, :]))[::-1]

    # Make image pyramids
    channel_axis = {}
    if len(A.shape) > 2:
        channel_axis = {"channel_axis":-1} # pyramic_gaussian needs this parameter for color images
    AL  = tuple(pyramid_gaussian(A,  NLevels, downscale = 2, **channel_axis))
    ApL = tuple(pyramid_gaussian(Ap, NLevels, downscale = 2, **channel_axis))
    BL  = tuple(pyramid_gaussian(B,  NLevels, downscale = 2, **channel_axis))
    if verbose:
        print("AL:")
        for i in range(len(AL)):
            print(AL[i].shape)
    ApLColor = [img for img in ApL]
    if len(A.shape) > 2: # For color images, use grayscale features, as recommended in the paper
        AL  = [rgb2gray(img) for img in AL]
        ApL = [rgb2gray(img) for img in ApL]
        BL  = [rgb2gray(img) for img in BL]

    BpL = [] # B' values in pyramid (grayscale)
    BpLColor = [] # B' values in pyramid (color)
    BpLidx = [] # Indices of nearest neighbors at each pixel in pyramid
    if verbose:
        print("BL:")
    for i in range(len(BL)):
        if verbose:
            print(BL[i].shape)
        BpL.append(np.zeros(BL[i].shape))
        shape = list(BL[i].shape)
        if len(A.shape) > 2:
            shape.append(A.shape[2])
        BpLColor.append(np.zeros(shape))
        BpLidx.append(-1*np.ones((BL[i].shape[0], BL[i].shape[1], 2), dtype=int))
    
    if debug_images: 
        plt.figure(figsize=(12, 6))

    # Do multiresolution synthesis
    for level in range(NLevels, -1, -1):
        if verbose:
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
        shape = AL[level].shape
        ipatch, jpatch = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        ipatch = ipatch.flatten()
        jpatch = jpatch.flatten()
        ## Step 1b: Get patches
        A1  = np.pad(AL[level], d)
        Ap1 = np.pad(ApL[level], d)
        APatches  = get_patches(A1, KSpatial, ipatch, jpatch)
        ApPatches = get_causal_patches(Ap1, KSpatial, ipatch, jpatch)
        X = np.concatenate((APatches, ApPatches), 1)
        A2 = np.array([])
        Ap2 = np.array([])
        if level < NLevels:
            ## Step 1c: Use multiresolution features if necessary
            A2  = np.pad(imresize( AL[level+1],  AL[level].shape), d)
            Ap2 = np.pad(imresize(ApL[level+1], ApL[level].shape), d)
            A2Patches  = get_patches(A2,  KSpatial, ipatch, jpatch)
            Ap2Patches = get_patches(Ap2, KSpatial, ipatch, jpatch)
            X = np.concatenate((X, A2Patches, Ap2Patches), 1)
        ## Step 1d: Setup nearest neighbor structures
        nn = NNClass(X)

        ## Step 2: Fill in the pixels in scanline order
        F   = np.zeros(sz)
        B1  = np.pad(BL[level], d)
        Bp1 = np.pad(BpL[level], d)
        B2  = np.array([])
        Bp2 = np.array([])
        if level < NLevels:
            B2  = np.pad(imresize( BL[level+1],  BL[level].shape), d)
            Bp2 = np.pad(imresize(BpL[level+1], BpL[level].shape), d)
        for i in tqdm(range(BpL[level].shape[0])):
            for j in range(BpL[level].shape[1]):
                # Make the feature at this pixel
                # Full patch B
                F[0:area] = B1[i:i+KSpatial, j:j+KSpatial].flatten()
                # Causal patch B'
                F[area:area+carea] = Bp1[i:i+KSpatial, j:j+KSpatial].flatten()[0:carea]
                if level < NLevels:
                    # Use multiresolution features
                    F[area+carea:area*2+carea] = B2[i:i+KSpatial, j:j+KSpatial].flatten()
                    F[area*2+carea:] = Bp2[i:i+KSpatial, j:j+KSpatial].flatten()
                # Find index of most closely matching feature point in A
                tic = time.time()
                idx, dist = nn_query(nn, F)
                #print(F, dist)
                total_time += time.time()-tic
                distSqr = dist**2
                idx = int(idx[0][0])
                idx = [ipatch[idx], jpatch[idx]]
                if Kappa > 0:
                    # Compare with coherent pixel
                    (idxc, distSqrc) = get_coherence_match(F, A1, Ap1, A2, Ap2, BpLidx[level], KSpatial, i, j) 
                    fac = 1 + Kappa*(2.0**(level - NLevels))
                    if distSqrc < distSqr*fac*fac:
                        idx = idxc
                BpLidx[level][i, j, :] = idx
                BpL[level][i, j] = ApL[level][idx[0], idx[1]]
                Bp1[i+d, j+d] = ApL[level][idx[0], idx[1]] # Also need to update padded image (sneaky bug!)
                BpLColor[level][i, j, ...] = ApLColor[level][idx[0], idx[1], :]
            if i%20 == 0 and debug_images:
                write_image(BpLColor[level], "%i.png"%level)
        if verbose:
            print("Time nearest neighbors:", total_time)
        if debug_images:
            plt.clf()
            plt.subplot(122)
            plt.imshow(BpLidx[level][:, :, 0], cmap = 'magma')
            plt.axis("off")
            plt.colorbar()
            plt.title("Y")
            plt.subplot(121)
            plt.imshow(BpLidx[level][:, :, 1], cmap = 'magma')
            plt.axis("off")
            plt.colorbar()
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
    parser.add_argument('--njobs', type=int, default=1, help="Number of parallel processes to use in exact nearest neighbor search (only relevant if ann is 0)")
    parser.add_argument('--debugImages', type=int, default=1, help="Whether to output all images in pyramid and chosen indices progressively as B' is being constructed")
    parser.add_argument('--ann', type=int, default=1, help="If 1, use approximate nearest neighbors (requires pynndescent).  If 0, revert to sklearn exact nearest neighbors")
    parser.add_argument('--verbose', type=int, default=0, help="If 1, print debugging info")
    opt = parser.parse_args()

    A = read_image(opt.A)
    Ap = read_image(opt.Ap)
    B = read_image(opt.B)
    Bp = imanalogy(A, Ap, B, Kappa=opt.Kappa, NLevels=opt.NLevels, KCoarse=opt.KCoarse, KFine=opt.KFine, n_jobs=opt.njobs, debug_images=bool(opt.debugImages), use_ann=(opt.ann == 1), verbose=(opt.verbose == 1))
    write_image(Bp, opt.Bp)
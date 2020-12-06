import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from scipy.sparse import linalg
from tqdm import tqdm
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray, gray2rgb
import matplotlib.pyplot as plt
import cv2

def get_neighbors(source, i, j):
    neighbors = []
    if(i+1 < np.shape(source)[0]):
        neighbors.append((i+1,j))
    if(i-1 >= 0):
        neighbors.append((i-1,j))
    if(j+1 < np.shape(source)[1]):
        neighbors.append((i,j+1))
    if(j-1 >= 0):
        neighbors.append((i,j-1))
    return neighbors

def build_arrays(source, mask, target, mask_points, mix = False, strength = 1):
    side = np.shape(mask)[0] * np.shape(mask)[1]

    A = sps.lil_matrix((side, side))
    b = np.zeros((side, 3))

    for i,index in tqdm(enumerate(mask_points)):
        neighbors = get_neighbors(source, index[0], index[1])
        value = np.zeros(3)

        for point in neighbors:
            A[i,i] += 1
            if(mix):
                sg = source[index[0], index[1]] - source[point] # source gradient
                tg = target[index[0], index[1]] - target[point] # target gradient
                
                # take the largest absolute value 
                value[0] += sg[0] if np.abs(strength * sg[0]) > np.abs(tg[0]) else tg[0]
                value[1] += sg[1] if np.abs(strength * sg[1]) > np.abs(tg[1]) else tg[1]
                value[2] += sg[2] if np.abs(strength * sg[2]) > np.abs(tg[2]) else tg[2]
            else:
                value += source[index[0], index[1]] - source[point]

            if point in mask_points: 
                j = mask_points.index(point)
                A[i,j] = -1
            else:
                b[i] += target[point]
        b[i] += value
    return A, b

def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def poisson_blend(source, mask, target, mix = False, strength = 1):
    """
    Performs Poisson blending. Source, mask, and target are all numpy arrays
    of the same shape (this is ensured by the fix_images function called in
    main.py).

    Args:
        source - np.array of source image
        mask   - np.array of binary mask
        target - np.array of target image

    Returns:
        np.array of Poisson blended image
    """
    mask_points = np.argwhere(mask)
    mask_points = list(map(tuple, mask_points))

    print(np.shape(mask_points))
    
    # Build Sparse
    A, b = build_arrays(source, mask, target, mask_points, mix, strength)
    A = A.tocsr()

    #regular solve to be clipped later
    r = linalg.cg(A, b[:,0])[0]
    g = linalg.cg(A, b[:,1])[0]
    b = linalg.cg(A, b[:,2])[0]
    
    # go through the mask points and assign the new intensity
    for i,index in enumerate(mask_points):
        target[index][0] = np.clip(r[i], 0.0, 1.0)
        target[index][1] = np.clip(g[i], 0.0, 1.0)
        target[index][2] = np.clip(b[i], 0.0, 1.0)

    return target

for s in range(3,4):
    source_path = './data/images/dress_' + str(i) + '.jpg'
    mask_path = './output/cloth-silhouettes/dress_' + str(i) + '_mask.png'
    target_path = './data/textures/texture-blue-jeans.jpg'
    
    source = cv2.imread(source_path, cv2.IMREAD_COLOR)
    target = cv2.imread(target_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # BGR -> RGB:
    source = source[..., ::-1]
    target = target[..., ::-1]

    lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # grey_source = rgb2gray(source)
    grey_source = gray2rgb(cl)
    
    # uint8 -> float32
    grey_source = grey_source.astype(np.float32) / 255.
    target = target.astype(np.float32) / 255.
    mask = np.round(mask.astype(np.float32) / 255.)

    image = poisson_blend(grey_source, mask, target, mix = True, strength = 1)

    output = (np.clip(image, 0., 1.) * 255.).astype(np.uint8)
    grey_source = (np.clip(grey_source, 0., 1.) * 255.).astype(np.uint8)

    mask_array = mask != 0
    source[mask_array] = output[mask_array]

    output_dir = './output'
    output_path = "%s/res_img%02d.jpg" % (output_dir, i)
    cv2.imwrite(output_path, source[..., ::-1])

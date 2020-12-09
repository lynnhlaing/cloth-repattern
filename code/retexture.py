import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import alpha_blend
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray, gray2rgb
import random
import math
import sys

def get_patch_to_insert_transfer(tilesize, overlapsize, to_fill, to_fill_mask, texture, alpha, source):
    '''
    Returns patch to insert for texture transfer. This patch is chosen based on how well it matches the
    existing texture, and its correspondence to the source image.

    Arguments:
    tilesize     - the size of the tile
    overlapsize  - the overlap between tiles
    to_fill      - the section of the output image that we want to fill
    to_fill_mask - a mask of which part of to_fill has already been filled in previous steps
    texture      - a sample of the texture that we are using to replicate the source
    source       - the section of the source that corresponds to the to_fill patch

    Output:
    A patch of texture insert into the output image
    '''
    
    # TODO: Implement this function
    # We can use the same minimum cut algorithm as before, but we also want our synthesized texture
    # to have the same intensity as the source image. See webpage and paper for more details.
    filter_window = (tilesize,tilesize, 3)
    output_shape = tuple(np.subtract(texture.shape, filter_window) + 1) + filter_window
    strides = texture.strides + texture.strides
    patches = np.lib.stride_tricks.as_strided(texture,output_shape,strides)[...,::1,::1,:,:].squeeze()

    fill_SSD = np.zeros((patches.shape[0],patches.shape[1]))
    # source_SSD = np.zeros((patches.shape[0],patches.shape[1]))
    # double overlap
    fill_SSD += np.einsum('ijklm,klm->ij', patches[:,:,:overlapsize,:overlapsize,:], to_fill[:overlapsize,:overlapsize,:])
    #left overlap
    fill_SSD += np.einsum('ijklm,klm->ij', patches[:,:,overlapsize:,:overlapsize,:], to_fill[overlapsize:,:overlapsize,:])
    #right overlap
    fill_SSD += np.einsum('ijklm,klm->ij', patches[:,:,:overlapsize,overlapsize:,:], to_fill[:overlapsize,overlapsize:,:])
    #center overlap
    #pretty sure we don't need to do this. because in the to_fill non overlap, the error is 0. 
    if np.max(to_fill[overlapsize:,overlapsize:,:]) != 0:
        fill_SSD += np.einsum('ijklm,klm->ij', patches[:,:,overlapsize:,overlapsize:,:], to_fill[overlapsize:,overlapsize:,:])
    fill_SSD += np.einsum('ijklm,klm->ij', patches, to_fill)
    fill_SSD *= -2
    #double overlap
    fill_SSD += np.einsum('ijklm, ijklm->ij', patches[:,:,:overlapsize,:overlapsize,:], patches[:,:,:overlapsize,:overlapsize,:])

    fill_SSD += np.einsum('ijk, ijk', to_fill[:overlapsize,:overlapsize,:], to_fill[:overlapsize,:overlapsize,:])

    #left overlap
    fill_SSD += np.einsum('ijklm, ijklm->ij', patches[:,:,overlapsize:,:overlapsize,:], patches[:,:,overlapsize:,:overlapsize,:])
    fill_SSD += np.einsum('ijk, ijk', to_fill[overlapsize:,:overlapsize,:], to_fill[overlapsize:,:overlapsize,:])
    #right overlap
    fill_SSD += np.einsum('ijklm, ijklm->ij', patches[:,:,:overlapsize,overlapsize:,:], patches[:,:,:overlapsize,overlapsize:,:])
    fill_SSD += np.einsum('ijk, ijk', to_fill[:overlapsize,overlapsize:,:], to_fill[:overlapsize,overlapsize:,:])
    #center overlap
    #pretty sure we don't need to do this. because in the to_fill non overlap, the error is 0. 
    if np.max(to_fill[overlapsize:,overlapsize:,:]) != 0:
        fill_SSD += np.einsum('ijklm, ijklm->ij', patches[:,:,overlapsize:,overlapsize:,:], patches[:,:,overlapsize:,overlapsize:,:])
        fill_SSD += np.einsum('ijk, ijk', to_fill[overlapsize:,overlapsize:,:], to_fill[overlapsize:,overlapsize:,:])
    fill_SSD += np.einsum('ijklm, ijklm->ij', patches, patches)
    fill_SSD += np.einsum('ijk, ijk', to_fill, to_fill)
    # fill_SSD *= alpha
    # print(patches.shape, source.shape)
    # source_SSD += np.einsum('ijklm,klm->ij', patches, source)
    # print(False in np.equal(np.einsum('ijklm,klm->ij', patches, source),np.tensordot(patches,source,axes=([2,3,4],[0,1,2]))))
    # source_SSD *= -2
    # source_SSD += np.einsum('ijklm, ijklm->ij', patches, patches)
    # print(np.allclose(np.multiply(patches,patches),np.einsum('ijklm, ijklm->ijklm', patches, patches)))
    # print(np.sum(np.multiply(patches,patches),axis=(2,3,4)))
    # print(np.sum(np.einsum('ijklm, ijklm->ijklm', patches, patches),axis=(2,3,4)))
    # source_SSD += np.einsum('ijk, ijk', source, source)
    # print(np.einsum('ijk, ijk', source, source)==np.sum(source**2))
    # print(np.min(source_SSD),np.max(source_SSD))

    # source_SSD *= (1-alpha)

    # final_SSD = np.add(fill_SSD,source_SSD)
    tol = 1.1
    min_error = np.min(fill_SSD[np.nonzero(fill_SSD)])
    if(min_error < 0):
        tol = 1
    toleranced = np.array(np.where(fill_SSD.flatten()<=min_error*tol))
    select = fill_SSD.flatten()[random.choice(toleranced.flatten())]
    tile_indices = np.array(np.where(fill_SSD==select)).flatten()

    new_patch = texture[tile_indices[0]:tile_indices[0]+tilesize, tile_indices[1]:tile_indices[1]+tilesize,:]

    horizontal_energy = np.sum((new_patch[:overlapsize,:,:] - to_fill[:overlapsize,:,:])**2, axis=2)
    vertical_energy = np.sum((new_patch[:,:overlapsize,:] - to_fill[:,:overlapsize,:])**2,axis=2)
    for i in range(1,tilesize):
        for j in range(0,overlapsize):
            horizontal_energy[j,i] += min(np.take(horizontal_energy[:,i-1],[j-1,j,j+1],mode='clip'))
            vertical_energy[i,j] += min(np.take(vertical_energy[i-1,:],[j-1,j,j+1],mode='clip'))
    #use argmin along axis to condense energy matrix into proper indices.
    #use np where here
    horizontal_seam = np.zeros(tilesize).astype(int)
    vertical_seam = np.zeros(tilesize).astype(int)
    seam_mask = np.zeros((tilesize,tilesize,3)).astype(bool)
    for i in range(tilesize-1,-1,-1):
        if i == tilesize-1:
            horizontal_seam[i] = np.argmin(horizontal_energy[:,i])
            vertical_seam[i] = np.argmin(vertical_energy[i,:])
        else:
            last_h_index = horizontal_seam[i+1]
            last_v_index = vertical_seam[i+1]
            if last_h_index == 0:
                horizontal_seam[i] = np.argmin(np.take(horizontal_energy[:,i],[last_h_index,last_h_index+1],mode='clip')) + last_h_index
            else:
                horizontal_seam[i] = np.argmin(np.take(horizontal_energy[:,i],[last_h_index-1,last_h_index,last_h_index+1],mode='clip')) + last_h_index -1
            if last_v_index == 0:
                vertical_seam[i] = np.argmin(np.take(vertical_energy[i,:],[last_v_index,last_v_index+1],mode='clip')) + last_v_index
            else:
                vertical_seam[i] = np.argmin(np.take(vertical_energy[i,:],[last_v_index-1,last_v_index,last_v_index+1],mode='clip')) + last_v_index -1
        seam_mask[:int(horizontal_seam[i]),i,:] = True
        seam_mask[i,:int(vertical_seam[i]),:] = True
    if np.max(to_fill) == 0:
        return new_patch
    if np.max(to_fill[0,overlapsize:,0]) == 0:
        seam_mask[:overlapsize,overlapsize:tilesize,:] = False
    if np.max(to_fill[overlapsize:,0,0]) == 0:
        seam_mask[overlapsize:tilesize,:overlapsize,:] = False

    masked_patch = new_patch.copy()
    masked_patch[seam_mask] = to_fill[seam_mask]

    return masked_patch

def synthesize_texture(source,mask,textures,tilesize,overlapsize,n_iter):
    #assert for texture equal to mask regions
    #list set the number of vals in mask
    #4 seam min cut. left right, top and bottom
    # min cut currently takes 0.5 seconds avg. Try to keep runtime of min cut only below two seconds.

    #will need to return a full texture in the shape of the mask.
    #automatically calculate overlap size to best account for retexturing
    #need to have a set amount of iterations
    # options for naive tiling or strict seperation.

    adjsize = tilesize - overlapsize
    outsize = (mask.shape[0],mask.shape[1])
    imout = np.zeros((math.ceil(outsize[0] / adjsize) * adjsize + overlapsize, math.ceil(outsize[1] / adjsize) * adjsize + overlapsize, source.shape[2]))

    for n in range(n_iter):
        # decrease tilesize for later runs
        if n > 0:
            tilesize = math.ceil(tilesize * 0.7)
            overlapsize = math.ceil(tilesize / 6)
            adjsize = tilesize - overlapsize

        # Gradually increase alpha from 0.1 to 0.9 throughout the iterations
        if n_iter > 1:
            alpha = 0.8 * ((n) / (n_iter-1)) + 0.1
        # catch div by zero if there is only one iteration
        else:
            alpha = 0.1

        imout_mask = np.zeros((imout.shape[0], imout.shape[1]), dtype=bool)

        # We made the output image slightly larger than the source 
        # so now we'll make the source the same size by padding

        source = np.pad(source, [(0,imout.shape[0]-source.shape[0]),(0,imout.shape[1]-source.shape[1]), (0,0)], mode='symmetric')
    
        # iterate over top left corner indices of tiles
        for y in range(0,imout.shape[0]-tilesize+1, adjsize):
            for x in range(0,imout.shape[1]-tilesize+1, adjsize):
                # patch we want to fill
                to_fill = imout[y:y+tilesize, x:x + tilesize]

                fill_check = np.sum(mask[y:y+tilesize, x:x + tilesize])
                
                if fill_check == 0:
                    continue
                most_common_texture = np.bincount(mask[y:y+tilesize, x:x + tilesize].flatten().astype(int))[1:]
                texture_index = int(np.argmax(most_common_texture))
                texture = np.array(textures[texture_index])
                
                # mask of what part has been filled
                to_fill_mask = imout_mask[y:y+tilesize, x:x+tilesize]

                # get the patch we want to insert
                patch_to_insert = get_patch_to_insert_transfer(tilesize, overlapsize, to_fill, to_fill_mask, texture, alpha, source[y:y + tilesize, x:x+tilesize])

                # update the image and the mask
                imout[y:y+tilesize, x:x+tilesize,:] = patch_to_insert
                imout_mask[y:y+tilesize,x:x+tilesize] = True
                # print(imout[y:y+tilesize, x:x+tilesize,:] )
                # if not quiet:
                cv2.imshow("Output Image", imout/255.0)
                cv2.waitKey(10)
        cv2.waitKey(100)
    cv2.waitKey(500)

    imout = imout[:outsize[0],:outsize[1]]
    # print(imout.shape)
    imout = np.multiply(imout, mask[:,:,np.newaxis])
    # cv2.imshow("last Image", imout/255.0)
    # cv2.waitKey(1000)

    return imout

def transfer_texture(source,mask,target):

    print(mask.shape)
    # BGR -> RGB:
    source_copy = source.copy()[..., ::-1]
    target = target[..., ::-1]
    mask = mask * 255.0

    lab = cv2.cvtColor(source_copy, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # grey_source = rgb2gray(source)
    grey_source = gray2rgb(cl)
    
    # uint8 -> float32
    grey_source = grey_source.astype(np.float32) / 255.
    target = target.astype(np.float32) / 255.
    mask = np.round(mask.astype(np.float32) / 255.)

    image = alpha_blend.poisson_blend(grey_source, mask, target, mix = True, strength = 1)

    output = (np.clip(image, 0., 1.) * 255.).astype(np.uint8)
    # cv2.imshow("Output2 Image", output/255.0)
    # cv2.waitKey(1000)
    grey_source = (np.clip(grey_source, 0., 1.) * 255.).astype(np.uint8)
    # cv2.imshow("grey Image", grey_source/255.0)
    # cv2.waitKey(1000)
    mask_array = mask != 0
    print(source_copy.shape)
    print(source.shape)
    
    print(output.shape)
    source_copy[mask_array] = output[mask_array]
    cv2.imshow("s Image", source/255.0)
    cv2.waitKey(1000)

    # output_dir = './output'
    # output_path = "%s/res_img%02d.jpg" % (output_dir, i)
    # cv2.imwrite(output_path, source[..., ::-1])

    return source_copy[..., ::-1]
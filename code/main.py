import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

import retexture
import silhouette
import utils

from tqdm import tqdm

IMAGE_PATH = './data/images'
TEXTURE_PATH = './data/textures'
MASK_PATH = './output/cloth-silhouettes'
SYNTH_PATH = './output/synthesized-silhouettes'
OUTPUT_PATH = './output/final-output'

def main():

    # ARGUMENT PARSING
    arg_parser = argparse.ArgumentParser(description="Cloth Repattern")
    arg_parser.add_argument("-i", "--img", type=str, default=IMAGE_PATH,
                            help='''Specify input clothing image with the proper filename formatting to use
                            (default: ./data/images; note: default will run on all images within ./data/images)''')
    arg_parser.add_argument("-t", "--texture", type=str, default=TEXTURE_PATH,
                            help='''Specify input texture image with the proper filename formatting to use
                            (default: ./data/images; note: default will run on all textures within ./data/textures)''')
    arg_parser.add_argument("-s", "--silent", action="store_true",
                            help='''Will stop matplotlib from displaying each result before
                            moving onto the next image.''')
    args = vars(arg_parser.parse_args())

    if args['img'] != IMAGE_PATH:
        if os.path.exists(args['img']):
            input_imgs = [args['img']]
        else: 
            print("Specified image does not exist: %s" % args['img'])
            sys.exit(1)
    else:
        input_imgs = [os.path.join(IMAGE_PATH,f) for f in os.listdir(IMAGE_PATH) if not f.startswith('.')]
        

    if args['texture'] != TEXTURE_PATH:
        if os.path.exists(args['texture']):
            input_textures = [args['texture']]
        else: 
            print("Specified texture does not exist: %s" % args['texture'])
            sys.exit(1)
    else:
        input_textures = [os.path.join(TEXTURE_PATH,f) for f in os.listdir(TEXTURE_PATH) if not f.startswith('.')]
        
    for img_path in input_imgs:
        img_name = os.path.splitext(os.path.split(img_path)[-1])[0]
        img = cv2.imread(img_path)
        
        # SILHOUETTE MASK GENERATION
        image_mask = silhouette.create_clothing_mask(img)
        if not args["silent"]:
            plt.imshow(cv2.cvtColor(image_mask*255, cv2.COLOR_BGR2RGB)); plt.show()

        cv2.imwrite(os.path.join(MASK_PATH, f'{img_name}_mask.png'),image_mask*255)

        # TEXTURE REPATTERNING
        # hyperparameters
        n_iterations=3
        tilesize=42
        overlapsize=7
        print(np.shape(image_mask))
        # print(np.nonzero(np.sum(image_mask,axis=0))[0][0],np.nonzero(np.sum(image_mask,axis=1))[0][0])
        # print(np.nonzero(np.sum(image_mask,axis=0))[0][-1],np.nonzero(np.sum(image_mask,axis=1))[0][-1])
        # outsize = (img.shape[0], img.shape[1])
        #60,84,240,628, potentially dont need this, just use the image mask as the out
        # print(outsize)
        for texture_path in input_textures:
            texture_name = os.path.splitext(os.path.split(texture_path)[-1])[0]
            textures = cv2.imread(texture_path)

            texture_fill = retexture.synthesize_texture(img,image_mask,textures,tilesize,overlapsize,3)
            print(texture_fill)
            cv2.imwrite(os.path.join(SYNTH_PATH, f'{img_name}_{texture_name}_synth_silhouette.png'),texture_fill)
            if not args["silent"]:
                plt.imshow(cv2.cvtColor((texture_fill).astype(np.uint8), cv2.COLOR_BGR2RGB)); plt.show()

        # # COMPOSITING

            retextured = retexture.transfer_texture(img, image_mask, texture_fill)
            if not args["silent"]:
                plt.imshow(cv2.cvtColor((retextured).astype(np.uint8), cv2.COLOR_BGR2RGB)); plt.show()
            cv2.imwrite(os.path.join(OUTPUT_PATH, f'{img_name}_{texture_name}.png'), retextured)

main()
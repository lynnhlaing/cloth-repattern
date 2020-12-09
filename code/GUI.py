from tkinter import *
import tkinter as tk
from tkinter import ttk, colorchooser, filedialog
from PIL import Image, ImageTk, ImageGrab
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import scipy.misc
import scipy.cluster
import binascii
import os

import retexture
import silhouette
import utils

# os.environ["PATH"] += ":/usr/local/bin:/usr/local/bin/gs"
IMAGE_PATH = './data/images'
TEXTURE_PATH = './data/textures'
MASK_PATH = './output/cloth-silhouettes'
SYNTH_PATH = './output/synthesized-silhouettes'
OUTPUT_PATH = './output/final-output'

class main:
    def __init__(self,master):
        self.master = master
        self.color_fg = 'white'
        self.color_bg = 'white'
        self.textures = []
        self.old_x = None
        self.old_y = None
        self.penwidth = 10
        self.text_var = StringVar()
        self.binary_mask = []

        self.source_img = []
        self.source_img_name = ''
        self.texture_imgs = []
        self.texture_img_names = ''
        self.multi_mask = []

        self.drawWidgets()
        # self.prepare_mask()
        # self.draw_mask()

        self.master.rowconfigure(1, weight=1)

        self.canvas.bind('<B1-Motion>',self.paint) #draw the line 
        self.canvas.bind('<ButtonRelease-1>',self.reset)

    def start_retexture(self):
        self.multi_mask = self.export_mask()
        # custom_binary_mask = (self.multi_mask != 0) #the 0/1 mask of the drawn parts
        # self.texture_imgs is a list of the np arrays 
        # self.source_img is the uploaded source img (np array)
        n_iterations=3
        tilesize=42
        overlapsize=7

        texture_fill = retexture.synthesize_texture(self.source_img,self.multi_mask,self.texture_imgs,tilesize,overlapsize,1)
        cv2.imwrite(os.path.join(SYNTH_PATH, f'{self.source_img_name}{self.texture_img_names}_synth_silhouette.png'),cv2.cvtColor(texture_fill.astype(np.uint8),cv2.COLOR_RGB2BGR))

        retextured = retexture.transfer_texture(self.source_img, self.multi_mask, texture_fill)
        cv2.imwrite(os.path.join(OUTPUT_PATH, f'{self.source_img_name}{self.texture_img_names}.png'), cv2.cvtColor(retextured.astype(np.uint8),cv2.COLOR_RGB2BGR))
        
        self.draw_retextured(cv2.cvtColor(retextured,cv2.COLOR_RGB2BGR))

    def export_mask(self):  #changing the background color canvas
        fileName = "./output/cloth-silhouettes/custom_mask"
        # save postscipt image 
        self.canvas.postscript(file = fileName + '.eps') 
        # use PIL to convert to PNG 
        img = Image.open(fileName + '.eps') 
        fig = img.convert('RGBA').copy()

        data = np.asarray(fig)
        red, green, blue, alpha = data.T

        mask = np.zeros_like(data)[:, :, 0]

        for i, (texture, val) in enumerate(self.textures):
            h = val.lstrip('#')
            rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            color_match = (red == rgb[0]) & (blue == rgb[2]) & (green == rgb[1])

            mask[color_match.T] = i + 1

        return mask # values 0 to num textures + 1
        # fig.save(fileName + '.png', lossless = True) # this saves the canvas as a png

    def change_color(self):
        self.color_fg = self.text_var.get()

    def common_color(self, im):
        NUM_CLUSTERS = 5

        ar = np.asarray(im)
        shape = ar.shape
        ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
        codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)

        vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
        counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

        index_max = scipy.argmax(counts)                    # find most frequent
        peak = codes[index_max]
        color = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
        if(len(color) > 6):
            return color[:-2]
        return color

    def new_texture(self):  #adding a new texture
        #TODO: keep a list of texture paths as they come in to use in synthesize_texture
        ifile = filedialog.askopenfile(mode='rb',title='Choose a file')
        path = Image.open(ifile).convert("RGB")
        self.texture_img_names = self.texture_img_names + f'_{os.path.splitext(os.path.split(ifile.name)[-1])[0]}'
        self.texture_imgs.append(np.asarray(path))
        path = path.crop([ 0, 0, 40, 40])

        color = "#" + self.common_color(path)
        self.text_var.set(color)
        self.change_color()

        texture = ImageTk.PhotoImage(path)

        self.textures.append((texture, color))
        self.draw_texture_objects()

    def upload_source(self):
        ifile = filedialog.askopenfile(mode='rb',title='Choose a source')
        path = Image.open(ifile).convert("RGB")
        img = np.asarray(path)
        self.source_img = img
        self.source_img_name = os.path.splitext(os.path.split(ifile.name)[-1])[0]
        self.binary_mask = silhouette.create_clothing_mask(img)
        self.prepare_mask()
        self.draw_mask()
        
    def draw_texture_objects(self):
        Label(self.controls, 
        text="Choose texture to apply: ",
        justify = LEFT,
        padx = 20).grid(row=6,column=0)

        row = 7
        for texture, val in self.textures:
            Radiobutton(self.controls, 
                text="■",
                fg = val,
                selectcolor = 'black',
                image=texture,
                indicatoron = 0,
                compound = LEFT,
                borderwidth = 10,
                variable=self.text_var, 
                command=self.change_color,
                value=val,
            ).grid(row=row, column=0)
            row += 1

    def prepare_mask(self):
        # img = Image.open(r"./output/cloth-silhouettes/dress_1_mask.png").convert('RGBA')
        # arr=np.array(np.asarray(img))
        # r,g,b,a = np.rollaxis(arr,axis=-1)  
        # mask=((r==255)&(g==255)&(b==255))
        # arr[mask,3]=0

        arr = np.zeros_like(self.binary_mask)
        arr = np.dstack((arr, arr, arr, np.ones_like(self.binary_mask) * 255))
        arr[(self.binary_mask == 1),3]=0

        img=Image.fromarray(arr,mode='RGBA')

        self.img = ImageTk.PhotoImage(img)

        self.canvas.config(width=self.img.width() - 8, height=self.img.height() - 8)
    
    def draw_mask(self):
        self.canvas_img = self.canvas.create_image(0, 0, anchor="nw", image=self.img)
    
    def draw_retextured(self, retexture):
        self.canvas.config(width=retexture.width() - 8, height=retexture.height() - 8)
        self.canvas_img = self.canvas.create_image(0, 0, anchor="nw", image=retexture)

    def paint(self,e):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x,self.old_y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=True)
            self.draw_mask()

        self.old_x = e.x
        self.old_y = e.y

    def reset(self,e):    #reseting or cleaning the canvas 
        self.old_x = None
        self.old_y = None      

    def changeW(self,e): #change Width of pen through slider
        self.penwidth = e
           

    def clear(self):
        self.canvas.delete(ALL)
        self.draw_mask()

    # def change_fg(self):  #changing the pen color
    #     self.color_fg=colorchooser.askcolor(color=self.color_fg)[1]

    def change_bg(self):  #changing the background color canvas
        self.color_bg=colorchooser.askcolor(color=self.color_bg)[1]
        self.canvas['bg'] = self.color_bg

    def drawWidgets(self):
        self.controls = Frame(self.master,padx = 5,pady = 5)
        Label(self.controls, text='Pen Width:',font=('arial 10')).grid(row=1,column=0, pady=10)
        self.slider = ttk.Scale(self.controls,from_= 5, to = 100,command=self.changeW,orient=HORIZONTAL)
        self.slider.set(self.penwidth)
        self.slider.grid(row=2,column=0,ipadx=30)
        

        # self.brush_color_button = Button(self.controls, text = "Brush Color", command=self.change_fg) 
        # self.brush_color_button.grid(row=1,column=0,ipadx=30, ipady=10)

        # self.bg_color_button = Button(self.controls, text = "Background Color", command=self.change_bg) 
        # self.bg_color_button.grid(row=2,column=1,ipadx=30, ipady=10)

        self.clear_button = Button(self.controls, text = "Clear", fg="red", command=self.clear) 
        self.clear_button.grid(row=0,column=0,ipadx=10, ipady=5, sticky=E)

        self.exit_button = Button(self.controls, text = "Exit", fg="gray", command=self.master.destroy) 
        self.exit_button.grid(row=0,column=1,ipadx=10, ipady=5)

        self.controls.pack(side=LEFT, anchor=NW)

        self.draw_texture_objects()
        
        self.canvas = Canvas(self.master,width=500,height=400,bg=self.color_bg)
        self.canvas.pack(fill=BOTH,expand=True)

        self.new_texture_button= Button(self.controls, text = "Upload Source", command=self.upload_source) 
        self.new_texture_button.grid(row=3,column=0,ipadx=10, ipady=5, pady = (25,0))

        self.new_texture_button= Button(self.controls, text = "Upload Texture", fg="green", command=self.new_texture) 
        self.new_texture_button.grid(row=4,column=0,ipadx=10, ipady=5, pady = 15)

        self.new_texture_button= Button(self.controls, text = "Synthesize Texture!", command=self.start_retexture) 
        self.new_texture_button.grid(row=5,column=0,ipadx=10, ipady=5, pady = (0,25))
        
        

if __name__ == '__main__':
    root = Tk()
    main(root)
    root.title('Application')
    root.mainloop()

#https://www.youtube.com/watch?v=kzp7-0EFrIg
#https://stackoverflow.com/questions/3241929/python-find-dominant-most-common-color-in-an-image
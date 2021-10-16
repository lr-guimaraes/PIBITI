from PIL import Image, ImageFilter
import numpy as np
import os

DATA_DIR = os.path.join('filtros','data')#local da imagem

#visualizar as imagens em uma tela
def show_vertical(im1,im2):
    im = Image.fromarray(np.vstack((np.array(im1),np.array(im2))))
    im.show()

img = Image.open(os.path.join(DATA_DIR, 'img3.jpeg'))
filtered = img.filter(ImageFilter.BLUR)
show_vertical(img,filtered)

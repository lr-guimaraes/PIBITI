import os
import numpy as np
from PIL import Image, ImageFilter
INPUT_DIR = "input"
OUTPUT_DIR = "output"

def in_file(filename):
    return os.path.join(INPUT_DIR, filename)

def out_file(filename):
    return os.path.join(OUTPUT_DIR, filename)


def show_vertical(im1, im2):
    '''Retorna as imagens concatenadas na vertical'''

    im = Image.fromarray(np.vstack((np.array(im1), np.array(im2))))
    im.show()
    return im

def show_horizontal(im1, im2):
    '''Retorna as imagens concatenadas na horizontal'''

    im = Image.fromarray(np.hstack((np.array(im1), np.array(im2))))
    im.show()
    return im

from utils import show_horizontal, show_vertical
from PIL import Image, ImageFilter
import os

INPUT_DIR = os.path.join('filtros','data')
OUTPUT_DIR = os.path.join('filtros','output')

def show_box_blur(filename,r =1):
    original = Image.open(os.path.join(INPUT_DIR,filename))
    filtered  = original.filter(ImageFilter.BoxBlur(r))

 
    original = Image.open(os.path.join(INPUT_DIR,filename))
    filtered = original.filter(ImageFilter.BoxBlur(r))

    #mostrar as imagens lado a lado
    show_horizontal(original,filtered)
    filtered.save(
        os.path.join(OUTPUT_DIR,
    '{}_boxblur{}.jpg'.format(filename[:filename.index('.')], r)
    )
)


if __name__ == '__main__':
    show_box_blur("img4.jpg",8)
from PIL import Image, ImageFilter
import os
from utils import show_horizontal, show_vertical, in_file, out_file

def show_edges(filename,direction = 'x', offset=0): # offset deslocamento do numero de pixels

    original = Image.open(in_file(filename)).convert('L')# L-> escala de cinza
    XSOBL = ImageFilter.Kernel((3,3),
                            [-1, 0, 1,
                            -2, 0, 2,
                            -1, 0, 1],1, offset)

    YSOBL = ImageFilter.Kernel((3,3),
                            [-1, -2, -1,
                            0, 0, 0,
                            1, 2, 1],1, offset)
  
    if direction == 'x':
        filtered = original.filter(XSOBL)
    elif direction == 'y':
        filtered = original.filter(YSOBL)
    else:
        pass

    #mostrar as imagens lado a lado
    show_horizontal(original,filtered)
    filtered.save(
       out_file(
    '{}_spbel_{}.jpg'.format(filename[:filename.index('.')], direction, offset)
    )
)


if __name__ == "__main__":
   show_edges('img5.png',input("direção x ou y: "),0)

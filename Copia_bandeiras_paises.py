from PIL import Image
import os
import math

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

# retorna o endereço relatrivo dentro da pasta 'input'
def in_path(filename):
    return os.path.join(INPUT_FOLDER, filename)

image= Image.new("RGB", (700,700),(255,255,0)) # Tamnnho; Cor

def triangle(size):
    WHITE = (255,255,255)
    BLACK = (0,0,0)
    image = Image.new ("RGB", (size,size), WHITE)

    for x in range(size):
        for y in range(size):
            if x<y :
                image.putpixel((x,y),BLACK) #modificar; Cor nova
    return image

def bandeira_fraça(height):
    width = 3*height//2
    WHITE = (255,255,255)
    RED = (239,65,53)
    BLUE = (0,85,164)
    image = Image.new ("RGB", (width,height), WHITE)

    offset = width//3
    for x in range(offset):
        for y in range(height):
            image.putpixel((x,y),BLUE)
            image.putpixel((x + 2*offset,y),RED)
    return image

def bandeira_japao(height):

    width = 3*height//2
    WHITE = (255,255,255)
    RED = (173,35,51)

    raio = 3*height//10
    c = (width//2,height//2)

    image = Image.new ("RGB", (width,height), WHITE)
    for x in (c[0]-raio,c[0] + raio):
        for y in (c[1]-raio, c[1] + raio):
           if (x-c[0]**2) + (y-c[1])**2 <= raio**2:
               image.putpixel((x,y),RED)
    return image



if __name__ == "__main__":
   
    t = triangle(700)
    t.show()

    b = bandeira_fraça(700)
    b.show()    

    j = bandeira_japao(700)
    j.show()

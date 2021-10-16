from PIL import Image
from utils import in_file, out_file

def grayscale_(colored):
    w, h =  colored.size #comprimento e altura
    img = Image.new("RGB",(w, h))

    for x in range(w):
        for y in range(h):
            pxl = colored.getpixel((x,y)) #pixel
            lum = (pxl[0]+ pxl[1] + pxl[2])//3 #iluminação
            img.putpixel((x,y), (lum,lum,lum))
    return img

if __name__ == "__main__":
    img = Image.open(in_file("img2.jpg"))
    print(img.getpixel((100,100)))
    print(img.getpixel((500,300)))
    print(img.getpixel((300,180)))

    colorida = Image.open(in_file("img2.jpg"))
    pb_img = grayscale_(colorida)
    pb_img.save(out_file("pb_img2.jpg"))

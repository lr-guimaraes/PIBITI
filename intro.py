from PIL import Image

image = Image.open("img2.jpg")
print(image.getpixel((500,500))) # printar imagem como duplas

image.show() #exibir a imagem
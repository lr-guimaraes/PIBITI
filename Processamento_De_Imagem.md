# O que é uma imagem

- No ponto de vista computacional é uma matriz de pixels cujo cada um usa uma cor e a soma deles forma a imagem
- O sensor irá medir as informações luminosas e ira conferter um uma imagem

from PIL import Image

image = Image.open("img2.jpg")
print(image.getpixel((500,500))) # (182, 199, 209) #RGB

image.show() #exibir a imagem

# Biblioteca

- Pode-se utilizar  a biblioteca Pillow para manipular, criar e editar imagens

# Filtro convulcional
## carcteristicas

- Linear: Modificação na imagem, no qual o calculo da modificação de um pixel sera a partir da combinação dos valores que estão na vizinhança do pixel
  - Combinação lintear (Multiplicações e somas)
    - Media ponderada do pixels da vizinhança
-  Escialmente invariante: 
   -  o mesmo comportamento em toda imagem
  
### A partir da analize A[1][1] "8"  Para filtro de borramento

    Imagem*Nuclo * 9^-1 = Filtro de Borramento ( biblioteca pilow BoxBlur())                   
 
  $\left[\begin{array}{ccc}
  3 & 3 & 4 & 5 & 7 & 8 \\
  2 & 8 & 3 & 4 & 7 & 8 \\
  3 & 4 & 3 & 6 & 7 & 9 \\ 
  4 & 5 & 5 & 6 & 9 & 8 \\
  5 & 4 & 0 & 7 & 0 & 0 \\
  \end{array}\right]$ * 
  $\left[\begin{array}{ccc}
  1 & 1 & 1\\ 
  1 & 1 & 1\\ 
  1 & 1 & 1\\ 
  \end{array}\right]$ * 9^-1 = $\left[\begin{array}{ccc}
  &  \\
  Filtro\\ 
  escolhido\\
  &  \\
\end{array}\right]$ 


  A[0][0] "3" é
 correspondente ao B[0][0] "1" 

  A[1][1] "8" é correspondente ao B[1][1] "1"

- procedimento para bordas 
  - colocar 0 ou valores proximos

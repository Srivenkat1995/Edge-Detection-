import cv2
import numpy as np 
import math

image = cv2.imread('task1.png',0)

#print(image)
pad_image = np.asarray(image)
rows,columns= image.shape

#print(rows,columns)

new_image = [[0 for x in range(columns + 2)] for y in range(rows + 2 )]
dummy_image_x = [[0 for x in range(columns)] for y in range(rows)]
dummy_image_y = [[0 for x in range(columns)] for y in range(rows)]
dummy_image_xy = [[0 for x in range(columns)] for y in range(rows)]

for i in range(rows+1):
    for j in range(columns+1):
        if i == 0 or i == rows+1:
            new_image[i][j] = 0
        elif j == 0 or j == columns+1:
            new_image[i][j] = 0
        else:
            new_image[i][j] = pad_image[i-1][j-1]    

gauss_blur_filter = [[0 for x in range(3)] for y in range(3)]

gauss_blur_filter[0][0] = 1/16 
gauss_blur_filter[0][1] = 1/8
gauss_blur_filter[0][2] = 1/16
gauss_blur_filter[1][0] = 1/8
gauss_blur_filter[1][1] = 1/4
gauss_blur_filter[1][2] = 1/8
gauss_blur_filter[2][0] = 1/16
gauss_blur_filter[2][1] = 1/8
gauss_blur_filter[2][2] = 1/16

for i in range(rows):
    for j in range(columns):
        dummy_image_x[i][j] = new_image[i][j] * gauss_blur_filter[0][0] + new_image[i][j+1] * gauss_blur_filter[0][1] + new_image[i][j+2] * gauss_blur_filter[0][2] + new_image[1+1][j] * gauss_blur_filter[1][0] + new_image[1+1][j+1] * gauss_blur_filter[1][1] + new_image[1+1][j+2] * gauss_blur_filter[1][2] + new_image[i+2][j] * gauss_blur_filter[2][0] + new_image[i+2][j+1] * gauss_blur_filter[2][1] + new_image[i+2][j+2] * gauss_blur_filter[2][2]

#edge_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)

for i in range(rows):
    for j in range(columns):
        dummy_image_x[i][j] = - new_image[i][j] + new_image[i][j+2] - 2 * new_image[i+1][j] + 2 * new_image[i+1][j+2] - new_image[i+2][j] + new_image[i+2][j+2]
maximum = 0
for i in range(rows):
    for j in range(columns):
        if maximum < dummy_image_x[i][j] :
           maximum = dummy_image_x[i][j]
           
minimum = -250
for i in range(rows):
    for j in range(columns):
        if minimum > dummy_image_x[i][j] :
            minimum = dummy_image_x[i][j]          

for i in range(rows):
    for j in range(columns):
        val = dummy_image_x[i][j]
        constant =(val - minimum) / (maximum - minimum)
        dummy_image_x[i][j] = constant

final_image_x = np.asarray(dummy_image_x)


#################################### Along y axis ##############################

for i in range(rows):
    for j in range(columns):
        dummy_image_y[i][j] = new_image[i][j] * gauss_blur_filter[0][0] + new_image[i][j+1] * gauss_blur_filter[0][1] + new_image[i][j+2] * gauss_blur_filter[0][2] + new_image[1+1][j] * gauss_blur_filter[1][0] + new_image[1+1][j+1] * gauss_blur_filter[1][1] + new_image[1+1][j+2] * gauss_blur_filter[1][2] + new_image[i+2][j] * gauss_blur_filter[2][0] + new_image[i+2][j+1] * gauss_blur_filter[2][1] + new_image[i+2][j+2] * gauss_blur_filter[2][2]

#edge_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)

for i in range(rows):
    for j in range(columns):
        dummy_image_y[i][j] = - new_image[i][j] - new_image[i][j+2] - 2 * new_image[i][j+1] + 2 * new_image[i+2][j+1] + new_image[i+2][j] + new_image[i+2][j+2]
maximum = 0
for i in range(rows):
    for j in range(columns):
        if maximum < dummy_image_y[i][j] :
           maximum = dummy_image_y[i][j]
           
minimum = -250
for i in range(rows):
    for j in range(columns):
        if minimum > dummy_image_y[i][j] :
            minimum = dummy_image_y[i][j]          

for i in range(rows):
    for j in range(columns):
        val = dummy_image_y[i][j]
        constant =(val - minimum) / (maximum - minimum)
        dummy_image_y[i][j] = constant

final_image_y = np.asarray(dummy_image_y)

################################################# Combining both Filters along x and y axis..##################################

for i in range(rows):
    for j in range(columns):
        dummy_image_xy[i][j] = math.sqrt(dummy_image_x[i][j] ** 2 + dummy_image_y[i][j] ** 2 )


final_image = np.asarray(dummy_image_xy)

####################################################################################################################

cv2.imwrite('alongxaxis.png',final_image_x * 255)
cv2.imwrite('alongyaxis.png',final_image_y * 255)
cv2.imwrite('question1output.png',final_image * 255)

cv2.imshow('task1',final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()





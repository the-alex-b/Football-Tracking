
import cv2 
import glob
import numpy as np

def twodvisualisation(warped_coords, i, pitch_x, pitch_y): 

    path = './input_footage/picture/PitchTemplate.png'
    img = cv2.imread(path,0)

    # rescaling
    scale = 7
    img_resized = cv2.resize(img,dsize=(scale*pitch_x, scale*pitch_y), interpolation=cv2.INTER_AREA)

    # Loop trough all warped coords and place a circle on blank top view.
    for person in warped_coords:
        cv2.circle(img_resized, (int(person[0])*scale, int(person[1])*scale), 5, (0,0,255), 5)
    save_fig(img_resized, i) 


def save_fig(img, i): 
    cv2.imwrite('./output_images/{}_2visualisation.jpg'.format(i), img)


def save_vid(): 

    img_array = []
    for filename in sorted(glob.glob('./output_images/*_2visualisation.jpg'),key = lambda x : int(x.split('_')[1].split('/')[1])):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    out = cv2.VideoWriter('2visualisation.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)    
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

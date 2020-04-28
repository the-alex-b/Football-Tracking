
import cv2 
import glob
import numpy as np

def twodvisualisation(detections,i): 

    path = './input_footage/picture/PitchTemplate.png'
    img = cv2.imread(path,0)
    text_displacement = 2
    template_h = 74
    template_w = 115
    
    # rescaling
    scale = 7
    img_resized = cv2.resize(img,dsize=(scale*template_w, scale*template_h), interpolation=cv2.INTER_AREA)
    # Loop trough all warped coords and place a circle on blank top view.
    for person in detections:
        cv2.circle(img_resized, (int(person[0])*scale, int(person[1])*scale), 5, (0,0,255), 5)
        if len(person)>2:
            cv2.putText(img_resized,('{:.0f}'.format(person[2])),
                    ((int(person[0])+text_displacement)*scale, int((person[1])+text_displacement)*scale),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0,0,255))
    save_fig(img_resized, i) 


def save_fig(img, i): 
    cv2.imwrite('./output_images/{}_2visualisation.jpg'.format(i), img)


def save_vid(): 

    img_array = []
    for filename in sorted(glob.glob('./output_images/*_2visualisation.jpg'),key = lambda x : int(x.split('_')[1].split('/')[1])):
        img = cv2.imread(filename)
        img2 = cv2.imread(filename.replace('2visualisation','original'))
        img_new = np.zeros((img.shape[0],img2.shape[1],img.shape[2]))
        img_new = np.uint8(img_new)
        img_new[:img.shape[0],:img.shape[1],:img.shape[2]] = img
        print(img_new.shape)
        print(img2.shape)

        res = np.vstack((img_new,img2))
        height, width, layers = res.shape
        size = (width,height)
        img_array.append(res)
        print(res.shape)
    
    out = cv2.VideoWriter('2visualisation.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)    
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

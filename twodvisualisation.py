
import cv2 
import glob
import numpy as np

def twodvisualisation(detections,labels,i,filename,detection_color=None): # (0,0,0)*len(detections)

    path = './input_footage/picture/PitchTemplate.png'
    img = cv2.imread(path,0)

    text_displacement = 1
    template_h = 74
    template_w = 115
    
    # rescaling
    scale = 7
    img_resized = cv2.resize(img,dsize=(scale*template_w, scale*template_h), interpolation=cv2.INTER_AREA)
    # Converting into color image - can't draw color on a gray scale image
    img_resized = img_resized[:, :, np.newaxis]
    img_resized = cv2.cvtColor(img_resized,cv2.COLOR_GRAY2RGB) 
    # Loop trough all warped coords and place a circle on blank top view.
    for j,person in enumerate(detections):
        dcolor = tuple([int(e) for e in tuple(detection_color[j])])
        cv2.circle(img_resized, (int(person[0])*scale, int(person[1])*scale), 5, dcolor, 5)
        cv2.putText(img_resized,('{:.0f}'.format(labels[j])),
                ((int(person[0])+text_displacement)*scale, 
                int((person[1])+text_displacement)*scale),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,0,255),
                2)
    save_fig(img_resized, i,filename) 


def save_fig(img, i, filename): 
    cv2.imwrite('./output_images/{}_2visualisation_{}.jpg'.format(i,filename), img)


def save_vid(modulo=1): 

    img_array = []

    for filename in sorted(glob.glob('./output_images/*2visualisation_coloring.jpg'),key = lambda x : int(x.split('_')[1].split('/')[1])):
        
        idx = int(filename.split('_')[1].split('/')[1])


    for i in range(idx): # quick max idx
        
        img = cv2.imread('./output_images/' + str(i) + '_2visualisation_coloring.jpg')
        img2 = cv2.imread('./output_images/' + str(i*modulo) + '_overlayed_image.jpg')
        img3 = cv2.imread('./output_images/' + str(i) + '_2visualisation_afterbasictrack.jpg')

        img13 = np.hstack((img,img3))

        # rescale - width (nr cols) should be the same 
        #img_new = np.zeros((img13.shape[0],img3.shape[1],img13.shape[2]))
        #img_new = np.uint8(img_new)
        #img_new[:img13.shape[0],:img13.shape[1],:img13.shape[2]] = img13

        try: 
            img_new = cv2.resize(img2, (img13.shape[1],img13.shape[0]), interpolation=cv2.INTER_CUBIC)
        except: 
            img_new = img13

        #img_new3 = np.zeros((img3.shape[0],img2.shape[1],img3.shape[2]))
        #img_new3 = np.uint8(img_new3)
        #img_new3[:img3.shape[0],:img3.shape[1],:img3.shape[2]] = img3
        #print(img_new.shape)
        #print(img13.shape)
        res = np.vstack((img13,img_new))
        x = res.shape[1] - 200
        y = res.shape[0] - 200
        res = cv2.putText(res,str(i*modulo), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (128,128,0))
        height, width, layers = res.shape
        size = (width,height)
        img_array.append(res)
        #print(res.shape)
    
    out = cv2.VideoWriter('colored_visualisation.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)    
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

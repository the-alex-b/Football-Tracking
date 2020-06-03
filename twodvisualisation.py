
import cv2 
import glob
import numpy as np

def twodvisualisation(detections,labels,i,filename,detection_color=(0,0,255)): # (0,0,0)*len(detections)

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
        # dcolor = tuple([int(e) for e in tuple(detection_color[j])])
        dcolor = detection_color
        cv2.circle(img_resized, (int(person[0])*scale, int(person[1])*scale), 5, dcolor, 5)
        # cv2.putText(img_resized,('{:.0f}'.format(labels[j])),
                # ((int(person[0])+text_displacement)*scale, 
                # int((person[1])+text_displacement)*scale),
                # cv2.FONT_HERSHEY_SIMPLEX,
                # 0.5,
                # (0,0,255),
                # 2)
    
    return img_resized
    # save_fig(img_resized, i,filename) 


def save_fig(img, i, filename): 
    cv2.imwrite('./output_images/full/{}_2visualisation_{}.jpg'.format(i,filename), img)


def save_vid(): 

    img_array = []
    for filename in sorted(glob.glob('./output_images/full/homography/*.jpg'),key = lambda x : int(x.split('/')[-1].split('.jpg')[0])):
        print(filename)
        img = cv2.imread(filename)
    #     # img2 = cv2.imread(filename.replace('2visualisation','original'))
    #     # img_new = np.zeros((img.shape[0],img2.shape[1],img.shape[2]))
    #     # img_new = np.uint8(img_new)
    #     # img_new[:img.shape[0],:img.shape[1],:img.shape[2]] = img
    #     # print(img_new.shape)
    #     # print(img2.shape)

        # res = np.vstack((img_new,img2))
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
        # print(res.shape)
    
    out = cv2.VideoWriter('./output_images/video/full_homography_overlay.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)    
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

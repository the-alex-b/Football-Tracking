
import cv2 

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

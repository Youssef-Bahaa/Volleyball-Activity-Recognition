import os
import cv2


class BoxInfo:
    def __init__(self,line):
        words = line.split()
        self.action = words.pop()
        words = [int(str) for str in words]

        self.player_ID = words[0]
        del words[0]

        x1 , y1, x2 , y2 , frame_ID , lost, grouping, generated = words

        self.box = x1 , y1 , x2 , y2
        self.frame_ID = frame_ID
        self.lost = lost
        self.grouping = grouping
        self.generated = generated


    def crop_from_frame(self,img_frame):
        x1, y1, x2, y2 = self.box
        self.crop = img_frame[y1:y2 , x1:x2 ]
        return self.crop

    def draw_box(self,img_frame):
        x1, y1, x2, y2 = self.box
        thickness = 2
        color = (0, 255, 0)
        cv2.rectangle(img_frame,(x1, y1), (x2, y2) , color , thickness)
        return img_frame

    def save_crop(self,img_frame , path = 'crops'):
        crop = self.crop_from_frame(img_frame)

        os.makedirs(path,exist_ok=True)
        file_name = f'player_{self.player_ID}_frame{self.frame_ID}.jpg'

        cv2.imwrite('file_name',crop)
        print(f'Image Saved : {file_name}')


if __name__ == "__main__":
    with open('anot_sample.txt' , 'r') as f:
        line = f.readline()

    l = BoxInfo(line)
    img  = cv2.imread('51725/51715.jpg')
    #cp = l.crop_from_frame(img)
    #cp_rec = l.draw_box(cp)

    rec = l.draw_box(img)

    cv2.imshow('imag' , rec)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #l.crop_from_frame()
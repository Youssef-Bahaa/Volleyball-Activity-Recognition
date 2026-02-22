import boxinfo
from boxinfo import BoxInfo
import os
import cv2


def load_frames_boxes(path):
    '''
    Mapping Player into 41 boxes
    Player_0:[box0,box1,box2,....,box11],
    Player_1:[box0,box1,box2,....,box11],
    ...,
    Player_11:[box0,box1,box2,....,box11]

    '''
    player_boxes = {i:[] for i in range(12)}
    frame_boxes = {}


    with open(path ,'r') as file:

        for line in file:
            player_box = BoxInfo(line)
            if player_box.player_ID > 11:
                continue

            player_boxes[player_box.player_ID].append(player_box)

            '''
            Now Mapping Frame_ID into {Boxes in certain frame}
            Frame_12342:[box0,box1,box2,....,box11] 
            Frame_12112:[box0,box1,box2,....,box11] 
            '''

            frame_boxes[player_box.frame_ID].append(player_box)

    return frame_boxes




def vis_clip(clip_path,annot_path):
    frames_boxes = load_frames_boxes(annot_path)
    font = cv2.FONT_HERSHEY_DUPLEX


    for frame_id , boxes_info in frames_boxes.items() :
        img_path = os.path.join(clip_path,f'{frame_id}.jpg')
        color = (0,255,0)
        thickness = 2
        img = cv2.imread(img_path)

        for box_info in boxes_info:
            x1, y1, x2, y2 = box_info.box

            cv2.rectangle(img, (x1,y1) ,
                                (x2, y2),color,thickness)
            cv2.putText(img,box_info.category,(x1,y1-10),font,0.6,color,2)

        cv2.imshow('Image',img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()



def load_video_annot(video_path):
    '''
    We will open the annotation text for video group activity
    it will contain the middle frame that only has group activity label
    annotaions.txt exist in videos/video_num/annotations.txt
    We will map Clip_ID to Group Activity
    '''

    clip_category = {}
    with open(os.path.join(video_path,'annotations.txt') ,'r') as file:
        for line in file:
            seq = line.split()
            frame_id = seq[0].replace('.jpg','')
            activity = seq[1]
            clip_category[frame_id] = activity

    return clip_category



def load_volleyball_dataset(videos_root, annot_root):
    videos_annot = {}

    for idx , video_dir in enumerate(sorted(os.listdir(videos_root))):
        clips_annot = {}
        video_dir_path = os.path.join(videos_root,video_dir)
        clips = os.listdir(video_dir_path)
        clips = clips.sort()
        Activities = load_video_annot(video_dir_path)

        print(f'Processing Video {video_dir} / {len(os.listdir(videos_root))}')

        for clip_dir in clips :
            clip_dir_path = os.path.join(video_dir_path,clip_dir)

            annotation_path = os.path.join(annot_root,video_dir , clip_dir,'annotations.txt')

            frame_boxes = load_frames_boxes(annotation_path)
            clip_activity = Activities[clip_dir]

            clips_annot[clip_dir] = {
                'category':clip_activity,
                'frame_boxes_dct':frame_boxes
            }

        videos_annot[video_dir] = clips_annot

    return videos_annot












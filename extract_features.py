import numpy as np
import os
import torch
import pickle
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

from annot_loader import load_frames_boxes


def check():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(torch.version.cuda)
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())



def prepare_model(image_level = False):
    if image_level:
        preprocess = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    else:
        preprocess = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    model = models.resnet50(pretrained = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    model = nn.Sequential(*list(model.children())[:-2])
    return model , preprocess



def extract_features(clip_dir_path, annot_file, output_file, model, process, image_level=False):
    frame_boxes = load_frames_boxes(annot_file)

    for frame_id , box in frame_boxes.items():
        img_pth = os.path.join(clip_dir_path,f'{frame_id}.jpg')
        image = Image.open(img_pth).convert('RGB')
        if image_level:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            processed_image = process(image).unsqueeze(0).to(device)
            dnn_repr = model(processed_image)
            dnn_repr = dnn_repr.view(1,-1)
        else:
            processed_images = []
            for box_info  in box:
                x1 , y1 , x2 , y2 = box_info.box
                cropped_image = image.crop((x1,y1,x2,y2))

                processed_images.append(process(cropped_image).unsqueeze(0))
                processed_images = torch.cat(processed_images)
                dnn_repr = model(processed_images)
                dnn_repr = dnn_repr.view(1, -1)
                dnn_repr = dnn_repr.view(len(processed_images), -1)





def divider():
    categories_dct = {
        'l-pass': 0,
        'r-pass': 1,
        'l-spike': 2,
        'r_spike': 3,
        'l_set': 4,
        'r_set': 5,
        'l_winpoint': 6,
        'r_winpoint': 7
    }

    train_ids = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31,32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
    val_ids = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28,30, 33, 46, 49, 51]
    test_ids = [4, 5, 9, 11, 14, 20, 21, 25, 29,34, 35, 37, 43, 44, 45, 47]

if __name__ == '__main__':

     check()
     #prepare_model()
 
     Image_level = True
     model , process = prepare_model(Image_level)
 
     video_pth = '/videos'
     annot_pth = '/volleyball_tracking_annotation'
     output_pth = '/features/image-level/resnet'
 
 
     videos_dirs = os.listdir(video_pth)
     videos_dirs.sort()
 
     for video_id , video_dir in enumerate(videos_dirs):
         video_dir_path = os.path.join(video_pth,video_dir)
         clips = os.listdir(video_dir_path)
         clips.sort()
 
         for clip_dir in clips:
             clip_dir_pth = os.path.join(video_dir_path , clip_dir)
             annot_file = os.path.join(annot_pth, video_dir, clip_dir, f'{clip_dir}.txt')
             output_file_path = os.path.join(output_pth,video_dir)
 
             os.makedirs(output_file_path,exist_ok=True)
             output_file = os.path.join(output_file_path, f'{clip_dir}.npy')
 
 
 
             extract_features(clip_dir_pth,annot_file,output_file,model,process,Image_level)






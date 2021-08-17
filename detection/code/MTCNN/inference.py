from __future__ import print_function
import os
import argparse
import time
import numpy as np
import cv2
from mtcnn import MTCNN


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.


#!python test_widerface.py --trained_model "C:/Users/Admin/Desktop/model_retinaface/Retinaface_model_v2/Resnet50_Final.pth" --network resnet50 --save_folder "C:/Users/Admin/Desktop/Face Detection Dataset/demo/" --dataset_folder "C:/Users/Admin/Desktop/Face Detection Dataset/retinaface/val/images/" --confidence_threshold 0.5 --save_image --cpu
#!python test_widerface.py --trained_model "C:/Users/Admin/Desktop/model_retinaface/Retinaface_model_v2/mobilenet0.25_Final.pth" --network mobile0.25 --save_folder "C:/Users/Admin/Desktop/Face Recognition Dataset/demo/" --dataset_folder "C:/Users/Admin/Desktop/Face Recognition Dataset/new_lfw" --confidence_threshold 0.5 --save_image --cpu



def arg_parse():
    parser = argparse.ArgumentParser(description='Retinaface')
    
    parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
    parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    
    args = arg_parse()

    detector = MTCNN()

    # testing dataset
    testset_folder = args.dataset_folder
    testset_list = os.path.join(args.dataset_folder, "wider_val.txt")

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    
    # testing begin
    for i, img_name in enumerate(test_dataset):
        image_path = os.path.join(testset_folder, img_name)
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)
        
        _t['forward_pass'].tic()
        detections = detector.detect_faces(img)  # forward pass
        _t['forward_pass'].toc()

        # --------------------------------------------------------------------
        save_name = args.save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if not os.path.exists(os.path.join(args.save_folder)):
            os.makedirs(os.path.join(args.save_folder))

        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time))

        min_conf = args.confidence_threshold
        for det in detections:
            if det['confidence'] >= min_conf:
                x, y, width, height = det['box']
                keypoints = det['keypoints']
                cv2.rectangle(img, (x,y), (x+width,y+height), (0,155,255), 2)
                cv2.circle(img, (keypoints['left_eye']), 2, (0,155,255), 4)
                cv2.circle(img, (keypoints['right_eye']), 2, (0,155,255), 4)
                cv2.circle(img, (keypoints['nose']), 2, (0,155,255), 4)
                cv2.circle(img, (keypoints['mouth_left']), 2, (0,155,255), 4)
                cv2.circle(img, (keypoints['mouth_right']), 2, (0,155,255), 4)

            if not os.path.exists(os.path.join(args.save_folder, os.path.dirname(img_name))):
                os.makedirs(os.path.join(args.save_folder, os.path.dirname(img_name)))

            name = os.path.join(args.save_folder, img_name)
            cv2.imwrite(name, img)
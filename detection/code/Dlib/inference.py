from __future__ import print_function
import os
import argparse
import time
import numpy as np
import cv2
import dlib


def convert_rect(rect):
    return [(rect.left())]

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((5, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 5):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

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
    parser = argparse.ArgumentParser(description='DLib')
    
    parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
    parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
    parser.add_argument('--type', default="hog", type=str, help='Type of Dlib (hog) (mmod)')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    
    args = arg_parse()

    if (args.type == "hog"):
        FaceDetector = dlib.get_frontal_face_detector()
    if (args.type == "mmod"):
        FaceDetector = dlib.cnn_face_detection_model_v1("detection/Dlib/mmod_human_face_detector.dat")

    predictor = dlib.shape_predictor("detection/Dlib/shape_predictor_5_face_landmarks.dat")

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
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # img = np.float32(img_raw)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        _t['forward_pass'].tic()
        faceRects = FaceDetector(img, 1)  # forward pass
        _t['forward_pass'].toc()

        # --------------------------------------------------------------------
        save_name = args.save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if not os.path.exists(os.path.join(args.save_folder)):
            os.makedirs(os.path.join(args.save_folder))

        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time))


        for rect in faceRects:
            if (args.type == "mmod"):
                rect = rect.rect

            shape = predictor(img, rect)
            shape = shape_to_np(shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = rect_to_bb(rect)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for (x, y) in shape:
                cv2.circle(img, (x, y), 2, (0, 0, 255), 4)

            if not os.path.exists(os.path.join(args.save_folder, os.path.dirname(img_name))):
                os.makedirs(os.path.join(args.save_folder, os.path.dirname(img_name)))

            name = os.path.join(args.save_folder, img_name)
            cv2.imwrite(name, img)
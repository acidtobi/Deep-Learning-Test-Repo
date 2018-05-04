import numpy as np
import cv2
from scipy.ndimage import imread
from scipy.misc import imresize
import glob
from PIL import Image
import os

RUN_UNATTENDED = True

def inverse_transform(orig_img, size):
    return int(size * (orig_img.shape[0] / 400))

# perl bin/createsamples.pl positives.txt negatives.txt samples_new 2500 "opencv_createsamples -bgcolor 0 -bgthresh 20 -maxxangle 1.1 -maxyangle 1.1 -maxzangle 2.0 -maxidev 40 -w 40 -h 40"
# opencv_traincascade -data classifier_new -vec samples_new.vec -bg negatives.txt -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 1700 -numNeg 5000 -w 40 -h 40 -mode ALL -precalcValBufSize 4096 -precalcIdxBufSize 4096 -featureType LBP
# opencv_traincascade -data classifier_new2 -vec samples_new2.vec -bg negatives.txt -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.4 -numPos 1000 -numNeg 2000 -w 40 -h 40 -mode ALL -precalcValBufSize 2048 -precalcIdxBufSize 2048 -featureType HAAR

casc = cv2.CascadeClassifier('%s/classifier/cascade.xml' % os.getcwd())

dirs = [os.path.basename(x)
        for x in glob.glob('extracted_by_haarcascade/*')
        if os.path.isdir(x)]

for filename in sorted(glob.glob('../images/*/*')):

    if not filename.strip() or os.path.isdir(filename):
        print(filename)
        continue

    img_class = os.path.basename(os.path.dirname(filename))

    ## skip entire class if directory already existed
    if img_class in dirs:
        continue

    try:
        os.mkdir("extracted_by_haarcascade/%s" % img_class)
    except FileExistsError:
        pass

    print(filename)
    try:
        orig_img = imread(filename)[...,::-1]
    except OSError as e:
        print(e)
        continue

    print("resizing")
    img = imresize(orig_img, 400 / orig_img.shape[0], interp='bicubic')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("detecting")
    d = casc.detectMultiScale(gray, 1.1, 20, maxSize=(200, 200))

    if not RUN_UNATTENDED:
        print("annotating")
        for (x,y,w,h) in d:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    for (x,y,w,h) in d:

        crop = orig_img[inverse_transform(orig_img, y):inverse_transform(orig_img, y+h),
               inverse_transform(orig_img, x):inverse_transform(orig_img, x+w)]

        if not RUN_UNATTENDED:
            pass
            #cv2.imshow('img', crop)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        else:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(crop)

            im.save("extracted_by_haarcascade/%s/img%s_%d_%d_%d_%d.jpg" %
                    (img_class, os.path.basename(filename), x, y, w, h))




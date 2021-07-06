import matplotlib.pyplot as plt
from os import listdir
from os import makedirs
from os import path
import numpy as np
import cv2

if not path.exists('Kvasir-SEG-TLT'):
    makedirs('Kvasir-SEG-TLT/images/train')
    makedirs('Kvasir-SEG-TLT/images/val')
    makedirs('Kvasir-SEG-TLT/masks/train')
    makedirs('Kvasir-SEG-TLT/masks/val')

path_m = 'Kvasir-SEG/masks/'
path_im = 'Kvasir-SEG/images/'
target_m = 'Kvasir-SEG-TLT/masks/'
target_im = 'Kvasir-SEG-TLT/images/'
fnames = [f for f in listdir(path_m) if '.jpg' in f]
split = 800
out_img_width=320
out_img_height=320

for i, f in enumerate(fnames):
    if i>=split:
        folder = 'val/'
    else:
        folder = 'train/'
    # generate fname
    fname_new = str(i+1)
    for j in range(4-len(fname_new)):
        fname_new = '0'+fname_new
    fname_new = fname_new+'.png'
    # save img
    im = plt.imread(path_im+f)
    im = cv2.resize(im, (out_img_width, out_img_height))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(target_im+folder+fname_new, im)
    # save mask
    m = plt.imread(path_m+f)
    m = cv2.resize(m, (out_img_width, out_img_height))
    m = np.where(m.mean(axis=2)>50, 1, 0)
    cv2.imwrite(target_m+folder+fname_new, m)

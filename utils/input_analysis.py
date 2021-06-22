
import os
from PIL import Image

from hyperparams import *





if __name__ == '__main__':

    gen_with_1 = 2032
    gen_with_2 = 1960
    gen_with_3 = 2576

    target_folder = '/home/ruben/PycharmProjects/ScoreCalcico/data/sc_run_resized'
    set_img_menos = set()
    # for filename in os.listdir(os.path.join('../', ORIGINAL_SC_DATASET, SC_MAS)):
    #     img_path = os.path.join('../', ORIGINAL_SC_DATASET, SC_MAS, filename)
    #     im = Image.open(img_path)
    #
    #     set_img_menos.add(im.size)
    #     w,h = im.size
    #     crop_size = int(abs((gen_with_2 - w)/2))
    #     if w == gen_with_1 or w == gen_with_3:
    #         cropped = im.crop((crop_size,0,w-crop_size,h))
    #         cropped.save(os.path.join(target_folder,SC_MAS,filename))
    #     else:
    #         im.save(os.path.join(target_folder, SC_MAS, filename))

    for root, subdirs, files in os.walk(target_folder):
        for file in files:
            im = Image.open(os.path.join(root,file))
            set_img_menos.add(im.size)
    print(set_img_menos)



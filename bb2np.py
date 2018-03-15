import numpy as np
import argparse
import os
from os import walk
from PIL import Image
import cv2 as cv

parser = argparse.ArgumentParser(description='Convert bbox format to yolo format.')
parser.add_argument('--labels', dest='label_path', default='Labels/',
                   help='Labels input directory path. End with /')
parser.add_argument('--output', dest='output_data_path', default='data.npz',
                   help='path file to save output .npz file')
parser.add_argument('--o', dest='output_path', default='data',
                   help='path file to save output .npz files')
parser.add_argument('--images', dest='image_path', default='Images/',
                   help='Images output directory path to save output file. End with /')
parser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to pascal_classes.txt',
    default=os.path.join('model_data', 'coco_classes.txt'))

def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

args = parser.parse_args()
imagePath = args.image_path
labelPath = args.label_path

def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

classes = read_classes(args.classes_path)
print('classes %s'%classes)

output_data_path = args.output_data_path
output_path = args.output_path

# array of [class left top right bottom]
boxes = []

# array of images
images = []

if not os.path.exists(output_path):
    os.makedirs(output_path)

for cls_id in range(len(classes)):
    txt_name_list = []
    for (dirpath, dirnames, filenames) in walk(labelPath):
        if (dirpath == (labelPath + '%03d'%cls_id)):
            txt_name_list.extend(filenames)

    if (len(txt_name_list) == 0):
        continue

    i = 0
    for txt_name in txt_name_list:
        """ Open input text files """
        if not txt_name.endswith('.txt'):
            continue

        txt_path = labelPath + '%03d'%cls_id + '/' + txt_name

        print("\n")
        print("Input:" + txt_path)

        nameFile = os.path.splitext(txt_name)[0]
        img_path = str('%s/%03d/%s.JPEG'%(imagePath, cls_id, nameFile))
        im = Image.open(img_path)
        imData = np.asarray(im)
        orig_size = np.array([im.width, im.height])
        orig_size = np.expand_dims(orig_size, axis=0)

        txt_file = open(txt_path, "r")
        lines = txt_file.read().split('\n')
        k = 0
        for line in lines:
            elems = line.split(' ')
            if(len(elems) == 4):
                # xmin = elems[0] #left
                # ymin = elems[1] #top
                # xmax = elems[2] #right
                # ymax = elems[3] #bottom
                cltrb = [cls_id] + elems

                cltrb = np.asarray(cltrb).astype(np.int)
                
                # Convert to x_center, y_center, box_width, box_height, class.
                box = cltrb.reshape((-1, 5))
                boxes_xy = 0.5 * (box[:, 3:5] + box[:, 1:3])
                boxes_wh = box[:, 3:5] - box[:, 1:3]
                boxes_xy = boxes_xy / orig_size
                boxes_wh = boxes_wh / orig_size
                box = np.concatenate((boxes_xy, boxes_wh, box[:, 0:1]), axis=1)

                print("Box: %s" % cltrb)
                print("Append to boxes: %s" % box)
                # boxes.append(cltrb)
                print("Append to images: %s" % img_path)
                #images.append(imData)
                out = "%s/%s_%s_%s.npz"%(output_path,cls_id,nameFile,k)
                print("Save to %s" % out)
                np.savez_compressed("%s" % out, images=imData, boxes=box)
                k += 1
                i += 1
                
#         if (i == 4): # test load
#             break

print('\ndone convert %s boxes' % i)
# print('saving to %s' % (output_data_path))
# np.savez_compressed(output_data_path, images=images, boxes=boxes)
print('done')

test = 1
if (test):
    print('show image for test result')
    data = np.load(out)
    Image.fromarray(data['images']).show()

import numpy as np
import argparse
import glob
from os import path, makedirs
from PIL import Image
from tqdm import tqdm
from lxml import etree

parser = argparse.ArgumentParser(description='Convert Pascal VOC format to yolo format.')
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
    default=path.join('model_data', 'coco_classes.txt'))


args = parser.parse_args()
image_path = args.image_path
label_path = args.label_path

def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


output_data_path = args.output_data_path
output_path = args.output_path

if not path.exists(output_path):
    makedirs(output_path)

def process(classes, image_dir, labels):
    print(classes)
    print(labels)
    for label_file in tqdm(labels):
        with open(label_file, 'r') as f:
            tree = etree.parse(f)
            root = tree.getroot()

            filename = root.find('filename').text
            name_file = path.basename(filename)

            img_path = path.abspath(image_dir + filename)
            if not path.exists(img_path):
                continue

            im = Image.open(img_path)
            imData = np.asarray(im)
            orig_size = np.array([im.width, im.height])
            orig_size = np.expand_dims(orig_size, axis=0)

            k = 0
            objects = root.findall('object')
            for obj in objects:
                name = obj.find('name').text
                boxid = classes.index(name) + 1

                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                xmax = float(bbox.find('xmax').text)
                ymin = float(bbox.find('ymin').text)
                ymax = float(bbox.find('ymax').text)

                cltrb = [boxid] + [xmin, ymin, xmax, ymax]
                cltrb = np.asarray(cltrb).astype(np.int)
                box = cltrb.reshape((-1, 5))
                boxes_xy = 0.5 * (box[:, 3:5] + box[:, 1:3])
                boxes_wh = box[:, 3:5] - box[:, 1:3]
                boxes_xy = boxes_xy / orig_size
                boxes_wh = boxes_wh / orig_size
                box = np.concatenate((boxes_xy, boxes_wh, box[:, 0:1]), axis=1)

                print("Box: %s" % cltrb)
                print("Append to boxes: %s" % box)
                print("Append to images: %s" % img_path)

                out = "%s/%s_%s_%s.npz"%(output_path, boxid, name_file, k)
                print("Save to %s" % out)
                np.savez_compressed("%s" % out, images=imData, boxes=box)
                k += 1


if path.isdir(image_path):
    classes = read_classes(args.classes_path)
    labels = glob.glob(label_path + '*.xml')
    process(classes, image_path, labels)

print('done')

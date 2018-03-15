import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yad2k.utils.utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import (yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body)
from yad2k.models.keras_yolo import (yolo_filter_boxes, yolo_boxes_to_corners, yolo_eval)

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")

print('class_names %s'%class_names)
print('anchors %s'%anchors)

def create_model(anchors, class_names, load_pretrained='model_data/yolo.h5'):

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(608, 608, 3))

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    # topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    model_body = yolo_model
    model_body.load_weights(load_pretrained)

    return model_body

def predict(image_file):
    sess = K.get_session()

    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.
    
    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """

    image, image_data = preprocess_image(image_file, model_image_size = (608, 608))
    scores, boxes, classes = yolo_eval(yolo_outputs, np.reshape(image.size[::-1], [2]).astype(np.float32))

    result = sess.run([scores, boxes, classes],feed_dict={
        yolo_model.input: image_data,
        K.learning_phase(): 0
    })
    print(result)
    out_boxes, out_scores, out_classes = result

    # Print predictions info
    print('Found %s boxes for %s'%(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

    imshow(image)
    plt.rcParams["figure.figsize"] = (20,20)
    plt.show()

    return out_scores, out_boxes, out_classes

weights_file = 'model_data/yolo.h5'
# weights_file = 'trained_stage_1.h5'
# weights_file = 'trained_stage_2.h5'
# weights_file = 'trained_stage_3.h5'
# weights_file = 'trained_stage_1_model.h5'
# weights_file = 'trained_stage_2_model.h5'
weights_file = 'trained_stage_3_model.h5'
weights_file = 'trained_stage_3_best.h5'
yolo_model = create_model(anchors, class_names, weights_file)

# for layer in yolo_model.layers:
#     weights = layer.get_weights()
#     print(weights)
# print(yolo_model.get_config())
yolo_model.summary()

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

predict("Images/000/0001.JPEG")
predict("Images/000/0002.JPEG")
predict("Images/000/0003.JPEG")
predict("Images/000/0363.JPEG")

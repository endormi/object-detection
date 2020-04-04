import cv2
import cvlib
from cvlib.object_detection import draw_bbox
import matplotlib.pyplot as plt


i = cv2.imread('images/fruits.jpg')
bbox, label, conf = cvlib.detect_common_objects(i)

"""
Real time object detection:
bbox, label, conf = cvlib.detect_common_objects(i, confidence=0.25, model='yolov3-tiny')
"""

output = draw_bbox(i, bbox, label, conf)

plt.imshow(output)
plt.show()

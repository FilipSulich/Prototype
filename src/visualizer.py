import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2


image_path_0 = "/Prototype/data/train/02/rgb/0000.png"
image_path_1 = "/Prototype/data/train/02/rgb/0001.png"

annotations_0 = [
    {"obj_id": 5, "obj_bb": [296, 96, 128, 141]},
    {"obj_id": 6, "obj_bb": [173, 194, 141, 107]},
    {"obj_id": 7, "obj_bb": [294, 230, 251, 231]},
]

annotations_1 = [
    {"obj_id": 5, "obj_bb": [308, 95, 123, 143]},
    {"obj_id": 6, "obj_bb": [174, 181, 143, 115]},
    {"obj_id": 7, "obj_bb": [286, 239, 252, 221]},
]

img0 = cv2.imread(image_path_0)
img1 = cv2.imread(image_path_1)

colors = [
    (0, 0, 255),   
    (0, 255, 0),   
    (255, 0, 0)    
]

def draw_boxes(image, annotations):
    for ann, color in zip(annotations, colors):
        x, y, w, h = ann["obj_bb"]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"ID: {ann['obj_id']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return image

img0 = draw_boxes(img0, annotations_0)
img1 = draw_boxes(img1, annotations_1)

cv2.imshow("Frame 0", img0)
cv2.imshow("Frame 1", img1)

cv2.waitKey(0)
cv2.destroyAllWindows()

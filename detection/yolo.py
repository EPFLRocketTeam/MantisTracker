import cv2
from detection.library.yolo import darknet

class Yolo:

  def __init__(self, config_path, data_path, weights_path):
    """
    Initialize a YOLO detection network 
    """
    network, class_names, class_colors = darknet.load_network(config_path, data_path, weights_path)
    self.network = network
    self.class_names = class_names
    self.class_colors = class_colors
    self.width = darknet.network_width(self.network)
    self.height = darknet.network_height(self.network)

  def detect(self, image):
    """
    Given an image, detect the rocket on it
    image: an OpenCV image

    return detections, image: the sorted list of detected boxes, the image with the boxes
    """
    # Creates an image compatible with the Darknet
    darknet_image = darknet.make_image(self.width, self.height, 3)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

    # return a sorted list of predictions
    detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=0.5)
    
    # detections = [(name, accuracy, box), ...]
    # resize the box to the orginal image size
    # TODO: establish a convention for the box coordinates
    h, w, c = image.shape
    x_ratio, y_ratio = w / self.width, h / self.height
    resized_detections = []
    for detection in detections:
      x, y = detection[2][0]*x_ratio, detection[2][1]*y_ratio
      w, h = detection[2][2]*x_ratio, detection[2][3]*y_ratio
      x, y = x - w/2, y - h/2
      resized_detections.append((detection[0], detection[1], (x, y, w, h)))


    # draw the detection
    image = darknet.draw_boxes(detections, image_resized, self.class_colors)
    # convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(detections)

    # Free (C implementation)
    darknet.free_image(darknet_image)

    return resized_detections, image


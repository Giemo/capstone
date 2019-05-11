import os

import cv2
import numpy as np
import tensorflow as tf
import rospy
import time

from styx_msgs.msg import TrafficLight

LABELS = [
    TrafficLight.UNKNOWN,
    TrafficLight.RED,
    TrafficLight.YELLOW,
    TrafficLight.GREEN,
    TrafficLight.UNKNOWN
]
LABELS_NAME = [
    "UNKNOWN", 
    "RED", 
    "YELLOW", 
    "GREEN", 
    "UNKNOWN"
]

class TLClassifier(object):
    def __init__(self):
        model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_inference_graph.pb")
        
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with open(model, 'rb') as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        self.image_tensor = graph.get_tensor_by_name("import/image_tensor:0")
        self.detection_boxes = graph.get_tensor_by_name("import/detection_boxes:0")
        self.detection_scores = graph.get_tensor_by_name("import/detection_scores:0")
        self.detection_classes = graph.get_tensor_by_name("import/detection_classes:0")
        self.num_detections = graph.get_tensor_by_name("import/num_detections:0")

        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        self.session = tf.Session(graph=graph, config=config)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        # preprocess image
        resized_image = cv2.resize(image, (300, 300))
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        np_image = rgb_image.astype(np.float32)

        num_detections, classes, scores, boxes = self.session.run([self.num_detections, self.detection_classes, self.detection_scores, self.detection_boxes],
                                                  feed_dict={self.image_tensor: np.expand_dims(np_image, axis=0)})

        num_detections = np.squeeze(num_detections).astype(np.uint32)
        classes = np.squeeze(classes)
        scores = np.squeeze(scores)
        boxes = np.squeeze(boxes)

        for i in range(num_detections):
            label_idx = classes[i]

            if scores[i] > 0.50:
                rospy.loginfo("%s" % (LABELS_NAME[int(label_idx)]))
                return LABELS[int(label_idx)]
        
        rospy.loginfo("UNKKNOWN")
        return TrafficLight.UNKNOWN
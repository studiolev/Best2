import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import os

class FurnitureDetector:
    def __init__(self):
        # Load the model from TensorFlow Hub
        self.model = hub.load('https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1')
        self.furniture_classes = [
            'Chair', 'Table', 'Sofa', 'Bed', 'Cabinet', 'Desk', 'Lamp', 'Mirror',
            'Shelf', 'Cupboard', 'Bench', 'Stool', 'Ottoman', 'Sideboard', 'Wardrobe'
        ]
        
    def detect_furniture(self, image, confidence_threshold=0.5):
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Preprocess image
            input_img = tf.convert_to_tensor(img_array)[tf.newaxis, ...]
            
            # Run detection
            results = self.model(input_img)
            
            # Process results
            boxes = results['detection_boxes'][0].numpy()
            scores = results['detection_scores'][0].numpy()
            classes = results['detection_class_entities'][0].numpy()
            
            # Filter for furniture classes and confidence threshold
            detections = []
            for i in range(len(scores)):
                if scores[i] >= confidence_threshold:
                    class_name = classes[i].decode('utf-8')
                    if class_name in self.furniture_classes:
                        detections.append({
                            'class': class_name,
                            'confidence': float(scores[i]),
                            'bbox': boxes[i].tolist()
                        })
            
            # Draw bounding boxes on image
            img_with_boxes = image.copy()
            for detection in detections:
                bbox = detection['bbox']
                # Convert normalized coordinates to pixel coordinates
                h, w = image.size
                x1, y1, x2, y2 = [
                    int(bbox[1] * w),
                    int(bbox[0] * h),
                    int(bbox[3] * w),
                    int(bbox[2] * h)
                ]
                
                # Draw rectangle and label
                from PIL import ImageDraw
                draw = ImageDraw.Draw(img_with_boxes)
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                draw.text((x1, y1-10), f"{detection['class']} ({detection['confidence']:.2f})", 
                         fill='red')
            
            return {
                'detections': detections,
                'image': img_with_boxes
            }
            
        except Exception as e:
            print(f"Error in detection: {str(e)}")
            return {
                'detections': [],
                'image': image
            } 
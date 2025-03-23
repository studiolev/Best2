from ultralytics import YOLO
import numpy as np
from PIL import Image
import os

class FurnitureDetector:
    def __init__(self):
        try:
            # Initialize YOLO model with a smaller configuration
            model_path = os.getenv('YOLO_MODEL_PATH', 'yolov8n.pt')
            self.model = YOLO(model_path)
            
            # Define furniture classes (you can modify this list based on your needs)
            self.furniture_classes = [
                'chair', 'couch', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush'
            ]
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            self.model = None
            self.furniture_classes = []
    
    def detect_furniture(self, image, confidence_threshold=0.5):
        """
        Detect furniture in the image
        Args:
            image: PIL Image or numpy array
            confidence_threshold: float between 0 and 1
        Returns:
            dict with detections and processed image
        """
        try:
            if self.model is None:
                return {
                    'detections': [],
                    'image': image
                }
            
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Run YOLO detection with optimized settings
            results = self.model(image, conf=confidence_threshold, iou=0.45, verbose=False)[0]
            
            # Process results
            detections = []
            for box in results.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                
                # Only include furniture classes
                if class_name.lower() in self.furniture_classes:
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
            
            # Create a copy of the image for drawing
            processed_image = Image.fromarray(image.copy())
            
            # Draw boxes using PIL instead of OpenCV
            from PIL import ImageDraw
            draw = ImageDraw.Draw(processed_image)
            
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                # Draw label
                label = f"{detection['class']} {detection['confidence']:.2f}"
                draw.text((x1, y1-10), label, fill=(0, 255, 0))
            
            return {
                'detections': detections,
                'image': processed_image
            }
        except Exception as e:
            print(f"Error in detect_furniture: {str(e)}")
            return {
                'detections': [],
                'image': image
            } 
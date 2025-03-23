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
            # Cadeiras
            'Chair', 'Armchair', 'Office chair', 'Dining chair', 'Stool', 'Bench',
            # Mesas
            'Table', 'Coffee table', 'Dining table', 'Desk', 'Side table', 'Console table',
            # Sofás
            'Sofa', 'Couch', 'Loveseat', 'Ottoman',
            # Camas
            'Bed', 'Bed frame', 'Mattress',
            # Armários
            'Cabinet', 'Wardrobe', 'Closet', 'Cupboard', 'Sideboard', 'Buffet',
            # Iluminação
            'Lamp', 'Ceiling light', 'Chandelier', 'Wall lamp',
            # Espelhos
            'Mirror', 'Wall mirror',
            # Prateleiras
            'Shelf', 'Bookshelf', 'Wall shelf',
            # Outros
            'Rug', 'Curtain', 'Pillow', 'Blanket', 'Cushion'
        ]
        
        # Dicionário de lojas de móveis em Portugal
        self.furniture_stores = {
            'Chair': [
                {'name': 'Zara Home', 'url': 'https://www.zarahome.com/pt/cadeiras-c1000000000000000000.html'},
                {'name': 'QuartoSala', 'url': 'https://www.quartosala.com/pt/cadeiras'},
                {'name': 'Area Store', 'url': 'https://areastore.com/cadeiras'},
                {'name': 'IKEA Portugal', 'url': 'https://www.ikea.com/pt/pt/cat/cadeiras-20661/'}
            ],
            'Table': [
                {'name': 'Zara Home', 'url': 'https://www.zarahome.com/pt/mesas-c1000000000000000000.html'},
                {'name': 'QuartoSala', 'url': 'https://www.quartosala.com/pt/mesas'},
                {'name': 'Area Store', 'url': 'https://areastore.com/mesas'},
                {'name': 'IKEA Portugal', 'url': 'https://www.ikea.com/pt/pt/cat/mesas-20662/'}
            ],
            'Sofa': [
                {'name': 'Zara Home', 'url': 'https://www.zarahome.com/pt/sofas-c1000000000000000000.html'},
                {'name': 'QuartoSala', 'url': 'https://www.quartosala.com/pt/sofas'},
                {'name': 'Area Store', 'url': 'https://areastore.com/sofas'},
                {'name': 'IKEA Portugal', 'url': 'https://www.ikea.com/pt/pt/cat/sofas-20663/'}
            ],
            'Bed': [
                {'name': 'Zara Home', 'url': 'https://www.zarahome.com/pt/camas-c1000000000000000000.html'},
                {'name': 'QuartoSala', 'url': 'https://www.quartosala.com/pt/camas'},
                {'name': 'Area Store', 'url': 'https://areastore.com/camas'},
                {'name': 'IKEA Portugal', 'url': 'https://www.ikea.com/pt/pt/cat/camas-20664/'}
            ],
            'Cabinet': [
                {'name': 'Zara Home', 'url': 'https://www.zarahome.com/pt/armarios-c1000000000000000000.html'},
                {'name': 'QuartoSala', 'url': 'https://www.quartosala.com/pt/armarios'},
                {'name': 'Area Store', 'url': 'https://areastore.com/armarios'},
                {'name': 'IKEA Portugal', 'url': 'https://www.ikea.com/pt/pt/cat/armarios-20665/'}
            ],
            'Lamp': [
                {'name': 'Zara Home', 'url': 'https://www.zarahome.com/pt/iluminacao-c1000000000000000000.html'},
                {'name': 'QuartoSala', 'url': 'https://www.quartosala.com/pt/iluminacao'},
                {'name': 'Area Store', 'url': 'https://areastore.com/iluminacao'},
                {'name': 'IKEA Portugal', 'url': 'https://www.ikea.com/pt/pt/cat/iluminacao-20666/'}
            ],
            'Mirror': [
                {'name': 'Zara Home', 'url': 'https://www.zarahome.com/pt/espelhos-c1000000000000000000.html'},
                {'name': 'QuartoSala', 'url': 'https://www.quartosala.com/pt/espelhos'},
                {'name': 'Area Store', 'url': 'https://areastore.com/espelhos'},
                {'name': 'IKEA Portugal', 'url': 'https://www.ikea.com/pt/pt/cat/espelhos-20667/'}
            ],
            'Shelf': [
                {'name': 'Zara Home', 'url': 'https://www.zarahome.com/pt/prateleiras-c1000000000000000000.html'},
                {'name': 'QuartoSala', 'url': 'https://www.quartosala.com/pt/prateleiras'},
                {'name': 'Area Store', 'url': 'https://areastore.com/prateleiras'},
                {'name': 'IKEA Portugal', 'url': 'https://www.ikea.com/pt/pt/cat/prateleiras-20668/'}
            ],
            'Rug': [
                {'name': 'Zara Home', 'url': 'https://www.zarahome.com/pt/tapetes-c1000000000000000000.html'},
                {'name': 'QuartoSala', 'url': 'https://www.quartosala.com/pt/tapetes'},
                {'name': 'Area Store', 'url': 'https://areastore.com/tapetes'}
            ],
            'Curtain': [
                {'name': 'Zara Home', 'url': 'https://www.zarahome.com/pt/cortinas-c1000000000000000000.html'},
                {'name': 'QuartoSala', 'url': 'https://www.quartosala.com/pt/cortinas'},
                {'name': 'Area Store', 'url': 'https://areastore.com/cortinas'}
            ],
            'Pillow': [
                {'name': 'Zara Home', 'url': 'https://www.zarahome.com/pt/almofadas-c1000000000000000000.html'},
                {'name': 'QuartoSala', 'url': 'https://www.quartosala.com/pt/almofadas'},
                {'name': 'Area Store', 'url': 'https://areastore.com/almofadas'}
            ]
        }
        
    def get_similar_products(self, furniture_class):
        """Retorna links para produtos similares em lojas portuguesas"""
        # Encontra a classe base mais próxima
        base_class = None
        for base in self.furniture_stores.keys():
            if base.lower() in furniture_class.lower():
                base_class = base
                break
        
        if base_class and base_class in self.furniture_stores:
            return self.furniture_stores[base_class]
        return []
        
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
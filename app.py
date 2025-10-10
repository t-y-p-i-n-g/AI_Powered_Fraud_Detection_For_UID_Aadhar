from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
import cv2
import pytesseract
from PIL import Image
from PIL.ExifTags import TAGS
from ultralytics import YOLO
import os
import re
import numpy as np
from werkzeug.utils import secure_filename
import tempfile
import numpy as np
from huggingface_hub import hf_hub_download
from supervision import Detections
import requests
from skimage.metrics import structural_similarity as ssim
import xml.etree.ElementTree as ET
from pyzbar.pyzbar import decode
import supervision as sv


import firebase_admin
from firebase_admin import credentials, firestore

# Initializing Firebase 
try:
    cred = credentials.Certificate("firebase-credentials.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Successfully connected to Firebase.")
except Exception as e:
    print(f"Error connecting to Firebase: {e}")
    db = None

tesseract_path = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = tesseract_path

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

try:
    model = YOLO("yolov8n.pt")
except:
    model = None
    print("Warning: YOLO model not loaded. Install ultralytics and download yolov8n.pt")
    
    
# Loading the pre-trained YOLO Object Detection model
try:
    OBJECT_DETECTION_MODEL_PATH = "./models/best.pt"
    object_detection_model = YOLO(OBJECT_DETECTION_MODEL_PATH)
    print("Object detection model loaded successfully.")
    print(f"Object Detection Model classes: {object_detection_model.names}")
except Exception as e:
    object_detection_model = None
    print("Warning: YOLO model not loaded. Install ultralytics and download yolov8n.pt")
    print(f"Warning: Custom Object Detection model not loaded. Error: {e}")

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Cayley Table for Verhoeff Checksum
_d = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
)

# permutation table for Verhoeff Checksum
_p = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8)
)

_inv = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)

def generate_checksum(num_str):
    c = 0
    num_digits = [int(d) for d in num_str]
    for i, digit in enumerate(reversed(num_digits)):
        c = _d[c][_p[(i % 8)][digit]]
    return _inv[c]

def validate_checksum(num_str_with_checksum):
    c = 0
    num_digits = [int(d) for d in num_str_with_checksum]
    for i, digit in enumerate(reversed(num_digits)):
        c = _d[c][_p[(i % 8)][digit]]
    return c == 0

def get_exif_data(image_path):
    """Extract EXIF metadata from image"""
    try:
        image = Image.open(image_path)
        exif_data = {}
        if hasattr(image, '_getexif'):
            info = image._getexif()
            if info:
                for tag, value in info.items():
                    decoded = TAGS.get(tag, tag)
                    exif_data[decoded] = value
        return exif_data
    except Exception as e:
        return {"error": str(e)}
    
    
# Loading general object detection model (YOLO v8)
try:
    general_model = YOLO("yolov8n.pt")
except:
    general_model = None
    print("Warning: General YOLO model not loaded. Install ultralytics and download yolov8n.pt")

# Loading the pre-trained Aadhaar-specific YOLO model
repo_config = dict(
    repo_id="arnabdhar/YOLOv8-nano-aadhar-card",
    filename="model.pt",
    local_dir="./models"
)

def detect_objects_yolo(image_path):
    #Detecting objects in image using YOLO
    try:
        if general_model is None:
            return {"error": "YOLO model not available"}
        
        img = cv2.imread(image_path)
        results = general_model(img)  
        labels = [general_model.names[int(cls)] for cls in results[0].boxes.cls]
        
        human_detected = "person" in labels
        return {
            "detected_objects": labels,
            "human_detected": human_detected,
            "fraud_indicator": not human_detected if labels else False
        }
    except Exception as e:
        return {"error": str(e)}


# Loading the pre-trained YOLO Aadhar model
aadhaar_model = YOLO(hf_hub_download(**repo_config))
id2label = aadhaar_model.names
print(id2label)


# Verifying if the image is of frudulent Aadhar card or not using object detection
def run_object_verification(image_path, object_model_raw_results):
    if object_detection_model is None:
        return {"error": "Object detection model not available."}
    
    try:
        detected_objects = []
        is_tampered = False
        
        for box in object_model_raw_results.boxes:
            class_id = int(box.cls[0])
            class_name = object_detection_model.names[class_id]
            detected_objects.append(class_name)
            if class_name == 'Tampered': is_tampered = True

        return {
            "detected_objects": list(set(detected_objects)),
            "is_tampered": is_tampered
        }
    except Exception as e:
        return {"error": f"Object verification failed: {str(e)}"}



def decode_aadhaar_qr(image_path):
    try:
        image = Image.open(image_path)
        decoded_objects = decode(image)
        
        if not decoded_objects:
            return {"error": "QR Code not found or could not be read."}
            
        qr_data_raw = decoded_objects[0].data.decode('utf-8', errors='ignore')
        
        try:
            root = ET.fromstring(qr_data_raw)
            qr_attributes = root.attrib
            return {
                "name": qr_attributes.get("name"),
                "dob": qr_attributes.get("dob"),
                "gender": qr_attributes.get("gender"),
                "uid": qr_attributes.get("uid"),
            }
        except ET.ParseError:
            return {"error": "QR data is not valid XML.", "raw_data": qr_data_raw}
            
    except Exception as e:
        return {"error": f"QR Code processing failed: {str(e)}"}


def extract_aadhaar_data(image_path, text_model_raw_results):
    try:
        detections = Detections.from_ultralytics(text_model_raw_results)
        image = np.array(Image.open(image_path))
        aadhaar_data = {}
        key_mapping = {
            'NAME': 'Name',
            'AADHAR_NUMBER': 'Aadhaar Number',
            'GENDER': 'Gender',
            'DATE_OF_BIRTH': 'Date of Birth',
            'ADDRESS': 'Address'
        }
        
        for bbox, cls_name in zip(detections.xyxy, detections.data['class_name']):
            x1, y1, x2, y2 = map(int, bbox)
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # --- FIX: Define a custom config for Tesseract ---
            config = '--psm 6' # A good default for other fields
            cls_name_str = str(cls_name)
            
            if cls_name_str == 'AADHAR_NUMBER':
                # Use a specific, stricter config for the Aadhaar number
                config = '--psm 7 -c tessedit_char_whitelist=0123456789 '
            
            # OCR on the region of interest with the specified config
            text = pytesseract.image_to_string(
                roi,
                lang="eng+hin",
                config=config  # Apply the custom config here
            ).strip()
            
            normalized_key = key_mapping.get(cls_name_str, cls_name_str)
            aadhaar_data[normalized_key] = text
            
        print(f"Final Aadhaar Data: {aadhaar_data}")
        return aadhaar_data
    except Exception as e:
        return {"error": f"OCR failed: {str(e)}"}

def validate_aadhaar_number(aadhaar_data):
    # Validating Aadhaar number using Verhoeff checksum
    try:
        if not aadhaar_data.get("Aadhaar Number"):
            result =  {"valid": False, "reason": "No Aadhaar number found"}
            print(f"Validation result: {result}")
            return result
      
        clean_number = aadhaar_data["Aadhaar Number"].replace(" ", "")
        
        if len(clean_number) != 12:
            return {"valid": False, "reason": f"Invalid length: {len(clean_number)} (should be 12)"}
        
    
        if not clean_number.isdigit():
            return {"valid": False, "reason": "Contains non-digit characters"}
        
      
        is_valid = validate_checksum(clean_number)
        
        return {
            "valid": is_valid,
            "reason": "Valid Aadhaar number" if is_valid else "Invalid checksum",
            "clean_number": clean_number
        }
    except Exception as e:
        result = {"valid":False, "reason":f"Error: {str(e)}"}
        print(f"Validation error: {result}")
        return result
    

def create_annotated_image(image_path, text_model_results, object_model_results):
    try:
        image = cv2.imread(image_path)
        
        # Annotations from the text extraction model (Blue boxes)
        text_detections = Detections.from_ultralytics(text_model_results)
        text_box_annotator = sv.BoxAnnotator(color=sv.Color.BLUE, thickness=2)
        text_label_annotator = sv.LabelAnnotator(color=sv.Color.BLUE, text_color=sv.Color.WHITE, text_scale=0.5)
        
        image = text_box_annotator.annotate(scene=image.copy(), detections=text_detections)
        image = text_label_annotator.annotate(scene=image, detections=text_detections)

        # Annotations from your custom object verification model (Red boxes)
        object_detections = Detections.from_ultralytics(object_model_results)
        object_box_annotator = sv.BoxAnnotator(color=sv.Color.RED, thickness=2)
        object_label_annotator = sv.LabelAnnotator(color=sv.Color.RED, text_color=sv.Color.WHITE, text_scale=0.5)

        image = object_box_annotator.annotate(scene=image, detections=object_detections)
        image = object_label_annotator.annotate(scene=image, detections=object_detections)
        
        # Saving the annotated image to the static folder
        annotated_filename = "annotated_" + os.path.basename(image_path)
        save_path = os.path.join('static', annotated_filename)
        cv2.imwrite(save_path, image)
        
        return annotated_filename
    except Exception as e:
        print(f"Error creating annotated image: {e}")
        return None

        
def analyze_aadhar_pair(front_path, back_path):
    # running the text extaction model on both front and back images
    text_model_raw_results_front = aadhaar_model.predict(front_path, verbose=False)[0]
    text_model_raw_results_back = aadhaar_model.predict(back_path, verbose=False)[0]
    
    front_ocr_results = extract_aadhaar_data(front_path, text_model_raw_results_front)
    back_ocr_results = extract_aadhaar_data(back_path, text_model_raw_results_back)
    
        
    
    # running tamper detection on front
    object_model_raw_results_front = object_detection_model(front_path, verbose=False)[0]
    object_results_front = run_object_verification(front_path, object_model_raw_results_front)
    
    # running tamper detection on back
    object_model_raw_results_back = object_detection_model(back_path, verbose=False)[0]
    object_results_back = run_object_verification(back_path, object_model_raw_results_back)
    
    # running exif on both front and back
    exif_results_front = get_exif_data(front_path)
    exif_results_back = get_exif_data(back_path)

    # qr code analysis
    qr_results = decode_aadhaar_qr(back_path)
    
    general_model_results_front = general_model(front_path, verbose=False)[0]
    
    general_labels = [general_model.names[int(cls)] for cls in general_model_results_front.boxes.cls]
    human_detected = "person" in general_labels
    results = {
        "front": {
            "object_verification": object_results_front,
            "exif_analysis": exif_results_front,
            "ocr_analysis": front_ocr_results,   
            "face_detection": {"human_detected": human_detected, "detected_objects": general_labels},
        },
        "back": {
            "object_verification": object_results_back,
            "exif_analysis": exif_results_back,
            "ocr_analysis": back_ocr_results,
            "qr_analysis": qr_results,
            "general_detection": {"human_detected":human_detected, "detected_objects": general_labels}   
        },
        "fraud_indicators": [],
        "raw_results": {
            "text_front": text_model_raw_results_front,
            "text_back": text_model_raw_results_back,
            "object_front": object_model_raw_results_front,
            "object_back": object_model_raw_results_back
        }
    }
    
    combined_ocr_results = front_ocr_results.copy()
    if "Address" in back_ocr_results:
        combined_ocr_results["Address"] = back_ocr_results["Address"]
    results['combined_ocr'] = combined_ocr_results
    
    
    fraud_score = 0
    
    if "error" not in object_results_front and object_results_front.get("is_tampered"):
        results["fraud_indicators"].append("Tampered region detected on the front of the card.")
        fraud_score += 3
        
    if not human_detected:
        results["fraud_indicators"].append("No human detected in the photo area (possible fake document).")
        fraud_score += 1

    if "error" not in combined_ocr_results and "error" not in qr_results:
        ocr_name = combined_ocr_results.get("Name", "").strip().lower()
        qr_name = qr_results.get("name", "").strip().lower()
        if ocr_name and qr_name and ocr_name not in qr_name:
             results["fraud_indicators"].append("Mismatch: Printed Name vs. QR Code Name.")
             fraud_score += 3
             
        # for gender
        ocr_gender = combined_ocr_results.get("Gender", "").strip.lower()
        qr_gender = qr_results.get("gender", "").strip().lower()
        if ocr_gender and qr_gender and ocr_gender not in qr_gender and qr_gender not in ocr_gender:
             results["fraud_indicators"].append("Mismatch between printed gender and QR code name.")
             fraud_score += 1
        
        # for dob
        ocr_dob_raw = combined_ocr_results.get("DATE_OF_BIRTH", "")
        ocr_dob = ocr_dob_raw.replace("-", "/").strip()
        qr_dob = qr_results.get("dob", "").replace("-","/").strip()
        if((ocr_dob and qr_dob) and (ocr_dob != qr_dob)):
             results["fraud_indicators"].append("Mismatch between printed date of birth (DOB) and QR date of birth (DOB).")
             fraud_score += 1
        
        # for aadhar number
        ocr_num = combined_ocr_results.get("AADHAR_NUMBER", "").strip().lower()
        qr_num = qr_results.get("aadhar_num", "").strip().lower()
        if ocr_num and qr_num and ocr_num not in qr_num and qr_num not in ocr_num:
             results["fraud_indicators"].append("Mismatch between printed date of birth (DOB) and QR date of birth (DOB).")
             fraud_score += 1
        
        # for address
        ocr_address = combined_ocr_results.get("ADDRESS", "").strip().lower()
        qr_address = qr_results.get("address", "").strip().lower()
        if ocr_address and qr_address and ocr_address not in qr_address and qr_address not in ocr_address:
             results["fraud_indicators"].append("Mismatch between printed address and QR code address.")
             fraud_score += 1


    if "error" not in results["front"]["ocr_analysis"]:
        results["aadhaar_validation"] = validate_aadhaar_number(results["front"]["ocr_analysis"])

    
    # Check object detection for fraud indicators
    if "error" not in object_results_front:
        if object_results_front.get("fraud_indicator"):
            results["fraud_indicators"].append("No human detected in image (possible fake document)")
            fraud_score += 1

    if(("error" not in exif_results_front or len(results["exif_analysis_front"]) == 0) or ("error" not in exif_results_back or len(results["exif_analysis_back"]) == 0)):
        results["fraud_indicators"].append("No EXIF metadata found")
        fraud_score += 0

    if "aadhaar_validation" in results and not results["aadhaar_validation"]["valid"]:
        results["fraud_indicators"].append(
            f"Invalid Aadhaar number: {results['aadhaar_validation']['reason']}"
        )
        fraud_score += 2

    results["fraud_score"] = fraud_score
    results["assessment"] = (
        "HIGH FRAUD RISK" if fraud_score >= 3 else
        "MODERATE FRAUD RISK" if fraud_score >= 1 else
        "LOW FRAUD RISK"
    )
    
    # saving the results to the firebase db
    if db:
        try:
            results_for_firestore = results.copy()
            
            # removing the key that contains raw YOLO objects
            if 'raw_results' in results_for_firestore:
                del results_for_firestore['raw_results']
            
            results_for_firestore['timestamp'] = firestore.SERVER_TIMESTAMP
            
            db.collection('analyses').add(results_for_firestore)
            print("Analysis results saved to Firestore.")
        except Exception as e:
            print(f"Error saving to Firestore: {e}")

    return results


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'front_image' not in request.files or 'back_image' not in request.files:
        flash('Please upload both front and back of the Aadhar card')
        return redirect(request.url)
    
    front_file = request.files['front_image']
    back_file = request.files['back_image']
    
    if front_file.filename == '' or back_file.filename == '':
        flash('Either one or both images are missing')
        return redirect(request.url)
    
    if (front_file and allowed_file(front_file.filename)) and (back_file and allowed_file(back_file.filename)):
        filename = secure_filename(front_file.filename)
        
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(front_file.filename)[1]) as tmp_front:
            front_file.save(tmp_front.name)
            front_path = tmp_front.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(back_file.filename)[1]) as tmp_back:
            back_file.save(tmp_back.name)
            back_path = tmp_back.name
            
        
        try:
            # Analyzing the image
            analysis_results = analyze_aadhar_pair(front_path, back_path)
            
            # creating annotated images using raw results from the analysis
            raw = analysis_results['raw_results']
            annotated_image_filename_front = create_annotated_image(front_path, raw['text_front'], raw['object_front'])
            
            annotated_image_filename_back = create_annotated_image(back_path, raw['text_back'], raw['object_back'])
            
            return render_template('results.html', 
                                 results=analysis_results,
                                 filename=f"{front_file.filename} & {back_file.filename}",
                                 annotated_image_filename_front=annotated_image_filename_front,
                                 annotated_image_filename_back=annotated_image_filename_back)
        finally:
            try:
                os.unlink(front_path)
                os.unlink(back_path)
            except OSError:
                pass  
    else:
        flash('Invalid file type. Please upload an image file.')
        return redirect(request.url)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic access"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400
    
    filename = secure_filename(file.filename)
    
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
    tmp_file_path = tmp_file.name
    tmp_file.close()
    
    try:
        file.save(tmp_file_path)
        analysis_results = analyze_aadhar_pair(tmp_file_path)
        return jsonify(analysis_results)
    finally:
        try:
            os.unlink(tmp_file_path)
        except OSError:
            pass

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
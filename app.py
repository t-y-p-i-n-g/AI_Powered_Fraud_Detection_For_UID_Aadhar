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
from pyzbar.pyzbar import decode
from pyaadhaar.utils import isSecureQr
from pyaadhaar.decode import AadhaarSecureQr

import firebase_admin
from firebase_admin import credentials, firestore
import json

from dotenv import load_dotenv

load_dotenv()

# Initializing Firebase 
try:
    # Check for the Hugging Face secret first
    creds_json_str = os.getenv('FIREBASE_CREDENTIALS_JSON')
    if creds_json_str:
        print("Loading Firebase credentials from environment variable (for deployment).")
        creds_dict = json.loads(creds_json_str)
        cred = credentials.Certificate(creds_dict)
    else:
        # If not found, fall back to the local file path from .env
        local_creds_path = os.getenv('FIREBASE_CREDS_PATH')
        if local_creds_path and os.path.exists(local_creds_path):
            print(f"Loading Firebase credentials from local file: {local_creds_path}")
            cred = credentials.Certificate(local_creds_path)
        else:
            raise FileNotFoundError("Firebase credentials not found in environment or local .env file.")

    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Successfully connected to Firebase.")

except Exception as e:
    print(f"Error connecting to Firebase: {e}")
    db = None
    

# Initializing tesseract
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
        confidences = []
        
        for box in object_model_raw_results.boxes:
            class_id = int(box.cls[0])
            class_name = object_detection_model.names[class_id]
            confidence = float(box.conf[0])
            
            detected_objects.append(class_name)
            if class_name == 'Tampered': is_tampered = True
            confidences.append(confidence)

        return {
            "detected_objects": list(set(detected_objects)),
            "is_tampered": is_tampered,
            "confidences":confidences
        }
    except Exception as e:
        return {"error": f"Object verification failed: {str(e)}"}


#======================================================================================
def decode_aadhaar_qr(image_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        code = decode(gray)
        
        if not code:
            return {"error": "QR Code not found or could not be read"}
        
        qrData = code[0].data
        isSecureQR = (isSecureQr(qrData))

        if isSecureQR:
            secure_qr = AadhaarSecureQr(int(qrData))
            decoded_secure_qr_data = secure_qr.decodeddata()
            if decoded_secure_qr_data:
                return decoded_secure_qr_data
            else:
                return {"error": "QR Code could not be found or could not be read."}
        else:
            # handling the case when QR code is not secured => Old QR
            # isSecureQr is false
            return {"error": "The detected QR code is not a secure Aadhaar QR code."}

    except Exception as e:
        return {"error": str(e)}


#====================================================================================================
def extract_aadhaar_data(image_path, text_model_raw_results):
    try:
        detections = Detections.from_ultralytics(text_model_raw_results)
        image = np.array(Image.open(image_path))
        aadhaar_data = {}
        confidences = []
        
        key_mapping = {
            'NAME': 'Name',
            'AADHAR_NUMBER': 'Aadhaar Number',
            'GENDER': 'Gender',
            'DATE_OF_BIRTH': 'Date of Birth',
            'ADDRESS': 'Address'
        }
        
        # The loop will now run for ALL detected fields
        for bbox, conf, cls_name in zip(detections.xyxy, detections.confidence, detections.data['class_name']):
            confidences.append(float(conf))
            x1, y1, x2, y2 = map(int, bbox)
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            cls_name_str = str(cls_name)
            config = '--psm 7 -c tessedit_char_whitelist=0123456789 ' if cls_name_str == 'AADHAR_NUMBER' else '--psm 6'
            
            text = pytesseract.image_to_string(roi, lang="eng+hin", config=config).strip()
            
            normalized_key = key_mapping.get(cls_name_str, cls_name_str)
            aadhaar_data[normalized_key] = text
            
        print(f"Final Aadhaar Data: {aadhaar_data}")
        return {"data": aadhaar_data, "confidences": confidences}
        
    except Exception as e:
        return {"error": f"OCR failed: {str(e)}"}
    
    
#================================================================================================
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
    

#===================================================================================================
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
    
    front_ocr_data = front_ocr_results.get("data", {})
    back_ocr_data = back_ocr_results.get("data", {})    
    
    # running tamper detection on front
    object_model_raw_results_front = object_detection_model(front_path, verbose=False)[0]
    object_results_front = run_object_verification(front_path, object_model_raw_results_front)
    
    # running tamper detection on back
    object_model_raw_results_back = object_detection_model(back_path, verbose=False)[0]
    object_results_back = run_object_verification(back_path, object_model_raw_results_back)
    
    # confidence scores
    all_confidences = []
    if object_model_raw_results_front.boxes:
        all_confidences.extend(object_model_raw_results_front.boxes.conf.tolist())
    if object_model_raw_results_back.boxes:
        all_confidences.extend(object_model_raw_results_back.boxes.conf.tolist())
    if text_model_raw_results_front.boxes:
        all_confidences.extend(text_model_raw_results_front.boxes.conf.tolist())
    if text_model_raw_results_back.boxes:
        all_confidences.extend(text_model_raw_results_back.boxes.conf.tolist())
        
    # calculating average confidence
    average_confidence = np.mean(all_confidences) if all_confidences else 0.0
    
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
            "ocr_analysis": front_ocr_data,   
            "face_detection": {"human_detected": human_detected, "detected_objects": general_labels},
        },
        "back": {
            "object_verification": object_results_back,
            "exif_analysis": exif_results_back,
            "ocr_analysis": back_ocr_data,
            "qr_analysis": qr_results,
            "general_detection": {"human_detected":human_detected, "detected_objects": general_labels}   
        },
        "average_confidence": average_confidence,
        "fraud_indicators": [],
        "raw_results": {
            "text_front": text_model_raw_results_front,
            "text_back": text_model_raw_results_back,
            "object_front": object_model_raw_results_front,
            "object_back": object_model_raw_results_back
        }
    }
    
    combined_ocr_results = front_ocr_data.copy()
    if "Address" in back_ocr_data:
        combined_ocr_results["Address"] = back_ocr_data["Address"]
    results['combined_ocr'] = combined_ocr_results
    
    print("Combined OCR Results: ", combined_ocr_results)
    
    
    fraud_score = 0
    
    if "error" not in object_results_front and object_results_front.get("is_tampered"):
        results["fraud_indicators"].append("Tampered region detected on the front of the card.")
        fraud_score += 3
        
    if not human_detected:
        results["fraud_indicators"].append("No human detected in the photo area (possible fake document).")
        fraud_score += 1

    if isinstance(combined_ocr_results, dict) and isinstance(qr_results, dict) and "error" not in qr_results:
        ocr_name = combined_ocr_results.get("Name", "").strip().lower()
        qr_name = qr_results.get("name", "").strip().lower()
        if ocr_name and qr_name and ocr_name not in qr_name:
             results["fraud_indicators"].append("Mismatch: Printed Name vs. QR Code Name.")
             fraud_score += 3
             
        # for gender
        ocr_gender = combined_ocr_results.get("Gender", "").strip().lower()
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


def transform_results_for_template(results):   
    # --- Overall Assessment ---
    risk_level = results.get('assessment', 'UNKNOWN').replace(" FRAUD RISK", "")
    risk_score = int(results.get('fraud_score', 0) * 20) # Convert a score out of 5 to a percentage

    color_map = {
        'HIGH': 'border-red-500 text-red-900 bg-red-50',
        'MODERATE': 'border-amber-500 text-amber-900 bg-amber-50',
        'LOW': 'border-green-500 text-green-900 bg-green-50',
        'UNKNOWN': 'border-slate-500 text-slate-900 bg-slate-50'
    }
    
    # --- Fraud Indicators ---
    indicators = []
    for desc in results.get('fraud_indicators', []):
        severity = 'high' if "mismatch" in desc.lower() or "tampered" in desc.lower() else 'medium'
        badge_map = {'high': 'border-red-300 bg-red-100 text-red-800', 'medium': 'border-amber-300 bg-amber-100 text-amber-800'}
        indicators.append({
            "type": desc.split(':')[0],
            "severity": severity,
            "description": desc,
            "badge_class": badge_map.get(severity, '')
        })

    # --- Front Card OCR ---
    ocr_front_data = results.get('front', {}).get('ocr_analysis', {})
    ocr_front_list = [
        {"label": "Name", "value": ocr_front_data.get("Name", "N/A"), "icon": "M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"},
        {"label": "Date of Birth", "value": ocr_front_data.get("Date of Birth", "N/A"), "icon": "M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"},
        {"label": "Gender", "value": ocr_front_data.get("Gender", "N/A"), "icon": "M13 10V3L4 14h7v7l9-11h-7z"},
        {"label": "Aadhaar Number", "value": ocr_front_data.get("Aadhaar Number", "N/A"), "icon": "M12 11c0 3.517-1.009 6.799-2.753 9.571m-3.44-2.04l.054-.09A13.916 13.916 0 008 11a4 4 0 118 0c0 1.017-.07 2.019-.203 3m-2.118 6.844A21.88 21.88 0 0015.171 17m3.839 1.132c.645-1.026.977-2.19.977-3.418a8 8 0 10-15.828-1.55A8 8 0 004 12c0-4.418 3.582-8 8-8s8 3.582 8 8z"},
    ]

    # --- Back Card Data ---
    qr_analysis = results.get('back', {}).get('qr_analysis', {})
    qr_data_list = []
    print(qr_analysis)
    
    if isinstance(qr_analysis, dict) and 'error' not in qr_analysis:
        for key, val in qr_analysis.items():
            qr_data_list.append({"label": key.capitalize(), "value": val})
            
    # --- Metadata ---
    exif_analysis = results.get('front', {}).get('exif_analysis', {})
    metadata_fields = []
    if 'error' not in exif_analysis:
        for key, val in exif_analysis.items():
            metadata_fields.append({"label": str(key), "value": str(val), "warning": "Software" in str(key)})

    transformed_data = {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'confidence_score': int(results.get('average_confidence', 0)*100),
        'risk_color_class': color_map.get(risk_level, 'border-slate-500'),
        'fraud_indicators': indicators,
        'ocr_status': "Success" if 'error' not in ocr_front_data else "Failed",
        'ocr_message': f"{len(ocr_front_data)} fields extracted" if 'error' not in ocr_front_data else "Extraction failed",
        'qr_status': "Decoded" if 'error' not in qr_analysis else "Failed",
        'qr_message': "Data successfully parsed" if 'error' not in qr_analysis else qr_analysis.get('error', 'Unknown error'),
        'exif_status': "Found" if exif_analysis and 'error' not in exif_analysis else "Not Found",
        'exif_message': f"{len(exif_analysis)} fields found" if exif_analysis and 'error' not in exif_analysis else "No EXIF data",
        'ocr_front': ocr_front_list,
        'ocr_address': results.get('combined_ocr', {}).get('Address', 'N/A'),
        'qr_decode_status': "Success" if 'error' not in qr_analysis else "Failed",
        'qr_data_items': qr_data_list,
        'qr_mismatch': any("Mismatch" in indicator for indicator in results.get('fraud_indicators', [])),
        'metadata_warning': any("Software" in str(field.get('label', '')) for field in metadata_fields),
        'metadata_warning_title': "Editing Software Detected",
        'metadata_warning_text': "The image metadata contains tags indicating it was processed by editing software, which can be a sign of digital manipulation.",
        'metadata_fields_left': metadata_fields[::2], # Split fields into two columns
        'metadata_fields_right': metadata_fields[1::2]
    }
    return transformed_data


@app.route('/')
def home():
    return render_template('upload.html')

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
            # Running the full, complex analysis
            analysis_results = analyze_aadhar_pair(front_path, back_path)
            
            # Transforming the results as per the template
            template_data = transform_results_for_template(analysis_results)
            
            # Creating annotated images
            raw = analysis_results['raw_results']
            annotated_image_filename_front = create_annotated_image(front_path, raw['text_front'], raw['object_front'])
            annotated_image_filename_back = create_annotated_image(back_path, raw['text_back'], raw['object_back'])
            
            # adding annotated image paths to template data
            template_data['front_annotated_image'] = url_for('static', filename=annotated_image_filename_front) if annotated_image_filename_front else None
            template_data['back_annotated_image'] = url_for('static', filename=annotated_image_filename_back) if annotated_image_filename_back else None
            
            # render the results
            return render_template('results.html', **template_data)
        finally:
            try:
                os.unlink(front_path)
                os.unlink(back_path)
            except OSError:
                pass  
    else:
        flash('Invalid file type. Please upload an image file.')
        return redirect(request.url)
    
@app.route('/analyzing')
def analyzing():
    return render_template('analyzing.html')

@app.route('/analyze', methods=['POST'])
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
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
    
    
# # Calculating the Structural Similarity Index Measure (SSIM)

# # Since SSIM is very sensitive to scaling and alingment, we need to align the template image and the image given by the user
# def align_images(img1, img2, max_features=500, keep_percent=0.2):
#     # Aligning images using ORB feature matching and homography
#     orb = cv2.ORB.create(max_features)
#     kps1, descs1 = orb.detectAndCompute(img1, None)
#     kps2, descs2 = orb.detectAndCompute(img2, None)
    
#     if descs1 is None or descs2 is None:
#         raise ValueError(f"Could not find enough keypoints for alignment")
    
#     matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = matcher.match(descs1, descs2)
#     matches = sorted(matches, key= lambda x: x.distance)
#     keep = int(len(matches) * keep_percent)
#     matches = matches[:keep]
    
#     pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
#     pts2 = np.float32([kps2[m.queryIdx].pt for m in matches])
    
#     H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
#     h, w = img2.shape[:2]
#     aligned = cv2.warpPerspective(img1, H, (w, h))
#     return aligned

# def ssim_preprocess_image(image_path, gray = True):
#     img = cv2.imread(image_path)
    
#     if img is None:
#         raise ValueError(f"Could not read image from the source: {image_path}")
    
#     if gray:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     return img

# ROIS = {
#     "aadhaar_number": (0.25, 0.75, 0.75, 0.90),
#     "dob": (0.05, 0.50, 0.35, 0.60),
#     "gender": (0.40, 0.50, 0.55, 0.60),
#     "name": (0.05, 0.35, 0.60, 0.45),
#     "logo": (0.05, 0.05, 0.25, 0.20)
# }

# def extract_roi(img, roi_box):
#     h, w = img.shape[:2]
#     x1 = int(roi_box[0] * w)
#     y1 = int(roi_box[1] * h)
#     x2 = int(roi_box[2] * w)
#     y2 = int(roi_box[3] * h)
#     return img[y1:y2, x1:x2]

# def ssim_compare_with_temps(input_image_path, templates_dir="templates/aadhar_templates", threshold = 0.75):
#     input_img = ssim_preprocess_image(input_image_path)
    
#     results = {}
#     best_match = None
#     best_score = -1
    
#     for template_file in os.listdir(templates_dir):
#         template_path = os.path.join(templates_dir, template_file)
        
#         template_img = ssim_preprocess_image(template_path, gray=True)
        
#         try:
#             aligned_input = align_images(input_img, template_img)
#         except Exception as e:
#             print(f"Aligment failed for {template_img}: {e}")
        
#         roi_scores = {}
#         roi_ssims = []
        
#         for roi_name, roi_box in ROIS.items():
#             try:
#                 roi_input = extract_roi(aligned_input, roi_box)
#                 roi_template = extract_roi(template_img, roi_box)
                
#                 if roi_input.size == 0 or roi_template.size == 0:
#                     continue
                
#                 roi_score, _ = ssim(roi_input, roi_template, full=True)
#                 roi_scores[roi_name] = roi_score
#                 roi_ssims.append(roi_score)
#             except Exception as e:
#                 roi_scores[roi_name] = f"Error : {e}"
        
#         if roi_ssims:
#             avg_score = float(np.mean(roi_ssims))
#             results[template_file] = {"avg_score": avg_score, "roi_scores": roi_scores}
        
    
#         if avg_score > best_score:
#             best_score = avg_score
#             best_match = template_file
    
#     is_valid = best_score >= threshold
    
#     results_arr = [best_match, best_score, results, is_valid]
#     print(results_arr)
    
#     return {
#         "best_match_template": best_match,
#         "best_score": best_score,
#         "all_scores": results,
#         "validity": is_valid
#     }
    
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
def run_object_verification(image_path):
    if object_detection_model is None:
        return {"error": "Object detection model not available."}
    
    try:
        results = object_detection_model(image_path)
        detected_objects = []
        is_tampered = False
        
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = object_detection_model.names[class_id]
                detected_objects.append(class_name)
                if class_name == 'Tampered':
                    is_tampered = True

        return {
            "detected_objects": list(set(detected_objects)),
            "is_tampered": is_tampered
        }
    except Exception as e:
        return {"error": f"Object verification failed: {str(e)}"}



def decode_aadhaar_qr(image_path):
    #Decodes the QR code from an Aadhaar card image and parses the XML data.
    try:
        image = Image.open(image_path)
        decoded_objects = decode(image)
        
        if not decoded_objects:
            return {"error": "QR Code not found or could not be read."}
            
        # The data is in the first decoded object
        qr_data_raw = decoded_objects[0].data.decode('utf-8')
        
        # Parse the XML data
        root = ET.fromstring(qr_data_raw)
        
        # Extract attributes from the 'PrintLetterBarcodeData' tag
        qr_attributes = root.attrib
        
        return {
            "name": qr_attributes.get("name"),
            "dob": qr_attributes.get("dob"),
            "gender": qr_attributes.get("gender"),
            "uid": qr_attributes.get("uid"), # This is the Aadhaar Number
            "raw_xml": qr_data_raw # For debugging
        }
    except Exception as e:
        return {"error": f"QR Code processing failed: {str(e)}"}


def extract_aadhaar_data(image_path):
    try:
        # Running the detection using Aadhaar-specific YOLO model
        results = aadhaar_model.predict(image_path, verbose=False)[0]
        detections = Detections.from_ultralytics(results)
        
        image = np.array(Image.open(image_path))
        
        aadhaar_data = {}
        
        key_mapping = {
            'NAME': 'Name',
            'AADHAR_NUMBER': 'Aadhaar Number',
            'GENDER': 'Gender',
            'DATE_OF_BIRTH': 'Date of Birth'
        }
        
        print(f"Detected classes: {detections.data['class_name']}") 
        
        for bbox, cls_name in zip(detections.xyxy, detections.data['class_name']):
            x1, y1, x2, y2 = map(int, bbox) # bounding box coords
            roi = image[y1:y2, x1:x2]   # crop region from original image
            
            if roi.size == 0:
                continue    # skipping empty crops
        
            # OCR on region of interest (roi) 
            text = pytesseract.image_to_string(
                roi, lang="eng+hin"  
            ).strip()
            
            cls_name_str = str(cls_name)
            normalized_key = key_mapping.get(cls_name_str, cls_name_str)
            
            print(f"Original key: {cls_name_str} -> Normalized: {normalized_key} -> Text: {text}")
            
            aadhaar_data[normalized_key] = text
        
        print(f"Final Aadhar Data: {aadhaar_data}")
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
        
def analyze_aadhar_pair(front_path, back_path):
    # running the text extaction model on both front and back images
    
    front_ocr_results = extract_aadhaar_data(front_path)
    back_ocr_results = extract_aadhaar_data(back_path)
    combined_ocr_results = front_ocr_results
    if "Address" in back_ocr_results:
        combined_ocr_results["Address"] = back_ocr_results["Address"]
        
    
    # running tamper detection on front
    object_results_front = run_object_verification(front_path)
    
    # running tamper detection on back
    object_results_back = run_object_verification(back_path)
    
    # running exif on both front and back
    exif_results_front = get_exif_data(front_path)
    exif_results_back = get_exif_data(back_path)

    # qr code analysis
    qr_results = decode_aadhaar_qr(back_path)
    
    results = {
        "object_verification_front": object_results_front,
        "object_verification_back": object_results_back,
        "exif_analysis_front": exif_results_front,
        "exif_analysis_back": exif_results_back,
        "ocr_analysis": combined_ocr_results,
        "object_detection_front": detect_objects_yolo(front_path),  
        "object_detection_back": detect_objects_yolo(back_path),
        "qr_analysis": qr_results,
        "fraud_indicators": []
    }
    
    if "error" not in "object_detection_front":
        if "object_detection_front".get("is_tampered"):
            results["fraud_indicators"].append("Document marked as 'Tampered' by object detection model")
            fraud_score += 4

    if "error" not in results["ocr_analysis"]:
        results["aadhaar_validation"] = validate_aadhaar_number(results["ocr_analysis"])

    fraud_score = 0
    
    # Check object detection for fraud indicators
    if "error" not in results["object_detection"]:
        if results["object_detection"].get("fraud_indicator"):
            results["fraud_indicators"].append("No human detected in image (possible fake document)")
            fraud_score += 1

    if not results["exif_analysis"] or len(results["exif_analysis"]) == 0:
        results["fraud_indicators"].append("No EXIF metadata found (possible digital manipulation)")
        fraud_score += 1

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
            results_with_timestamp = results.copy() # Avoid modifying the original dict
            results_with_timestamp['timestamp'] = firestore.SERVER_TIMESTAMP
            
            db.collection('analyses').add(results_with_timestamp)
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
    
    front_file = request.files['font_image']
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
            
            return render_template('results.html', 
                                 results=analysis_results, 
                                 filename=filename)
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
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
    
    
# Calculating the Structural Similarity Index Measure (SSIM)

# Since SSIM is very sensitive to scaling and alingment, we need to align the template image and the image given by the user
def align_images(img1, img2, max_features=500, keep_percent=0.2):
    # Aligning images using ORB feature matching and homography
    orb = cv2.ORB.create(max_features)
    kps1, descs1 = orb.detectAndCompute(img1, None)
    kps2, descs2 = orb.detectAndCompute(img2, None)
    
    if descs1 is None or descs2 is None:
        raise ValueError(f"Could not find enough keypoints for alignment")
    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descs1, descs2)
    matches = sorted(matches, key= lambda x: x.distance)
    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]
    
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.queryIdx].pt for m in matches])
    
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    h, w = img2.shape[:2]
    aligned = cv2.warpPerspective(img1, H, (w, h))
    return aligned

def ssim_preprocess_image(image_path, gray = True):
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image from the source: {image_path}")
    
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img

ROIS = {
    "aadhaar_number": (0.25, 0.75, 0.75, 0.90),
    "dob": (0.05, 0.50, 0.35, 0.60),
    "gender": (0.40, 0.50, 0.55, 0.60),
    "name": (0.05, 0.35, 0.60, 0.45),
    "logo": (0.05, 0.05, 0.25, 0.20)
}

def extract_roi(img, roi_box):
    h, w = img.shape[:2]
    x1 = int(roi_box[0] * w)
    y1 = int(roi_box[1] * h)
    x2 = int(roi_box[2] * w)
    y2 = int(roi_box[3] * h)
    return img[y1:y2, x1:x2]

def ssim_compare_with_temps(input_image_path, templates_dir="templates/aadhar_templates", threshold = 0.75):
    input_img = ssim_preprocess_image(input_image_path)
    
    results = {}
    best_match = None
    best_score = -1
    
    for template_file in os.listdir(templates_dir):
        template_path = os.path.join(templates_dir, template_file)
        
        template_img = ssim_preprocess_image(template_path, gray=True)
        
        try:
            aligned_input = align_images(input_img, template_img)
        except Exception as e:
            print(f"Aligment failed for {template_img}: {e}")
        
        roi_scores = {}
        roi_ssims = []
        
        for roi_name, roi_box in ROIS.items():
            try:
                roi_input = extract_roi(aligned_input, roi_box)
                roi_template = extract_roi(template_img, roi_box)
                
                if roi_input.size == 0 or roi_template.size == 0:
                    continue
                
                roi_score, _ = ssim(roi_input, roi_template, full=True)
                roi_scores[roi_name] = roi_score
                roi_ssims.append(roi_score)
            except Exception as e:
                roi_scores[roi_name] = f"Error : {e}"
        
        if roi_ssims:
            avg_score = float(np.mean(roi_ssims))
            results[template_file] = {"avg_score": avg_score, "roi_scores": roi_scores}
        
    
        if avg_score > best_score:
            best_score = avg_score
            best_match = template_file
    
    is_valid = best_score >= threshold
    
    results_arr = [best_match, best_score, results, is_valid]
    print(results_arr)
    
    return {
        "best_match_template": best_match,
        "best_score": best_score,
        "all_scores": results,
        "validity": is_valid
    }
    
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
    #Detect objects in image using YOLO
    try:
        if general_model is None:  # Use general_model instead of model
            return {"error": "YOLO model not available"}
        
        img = cv2.imread(image_path)
        results = general_model(img)  # Use general_model
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
        
def analyze_image(image_path):
    results = {
        "exif_analysis": get_exif_data(image_path),
        "ssim_analysis": ssim_compare_with_temps(image_path),
        "ocr_analysis": extract_aadhaar_data(image_path),
        "object_detection": detect_objects_yolo(image_path),  # ADD THIS LINE
        "fraud_indicators": []
    }

    if "error" not in results["ocr_analysis"]:
        results["aadhaar_validation"] = validate_aadhaar_number(results["ocr_analysis"])

    fraud_score = 0
    
    # ssim results
    ssim_res = results["ssim_analysis"]
    if not ssim_res["validity"]:
        results["fraud_indicators"].append(
            f"Low SSIM Score ({ssim_res['best_score']:.2f} with templates → Possible Tampering)"
        )
        fraud_score += 2

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

    return results


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
        tmp_file_path = tmp_file.name
        tmp_file.close() 
        
        try:
            file.save(tmp_file_path)
            
            # Analyzing the image
            analysis_results = analyze_image(tmp_file_path)
            
            return render_template('results.html', 
                                 results=analysis_results, 
                                 filename=filename)
        finally:
        
            try:
                os.unlink(tmp_file_path)
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
        analysis_results = analyze_image(tmp_file_path)
        return jsonify(analysis_results)
    finally:
        try:
            os.unlink(tmp_file_path)
        except OSError:
            pass

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
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

# Verhoeff Checksum Tables
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

def detect_objects_yolo(image_path):
    """Detect objects in image using YOLO"""
    try:
        if model is None:
            return {"error": "YOLO model not available"}
        
        img = cv2.imread(image_path)
        results = model(img)
        labels = [model.names[int(cls)] for cls in results[0].boxes.cls]
        
        human_detected = "person" in labels
        return {
            "detected_objects": labels,
            "human_detected": human_detected,
            "fraud_indicator": not human_detected if labels else False
        }
    except Exception as e:
        return {"error": str(e)}


def preprocess_aadhaar_image(image_path: str) -> np.ndarray:
    """
    Preprocess Aadhaar card image to enhance OCR accuracy.
    Returns a cleaned, ready-to-use image for pytesseract.
    """
    # 1. Read image
    img = cv2.imread(image_path)

    # 2. Resize for better DPI (scale up small images)
    h, w = img.shape[:2]
    if h < 800:  # arbitrary threshold for low-res
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 3. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # 5. Thresholding (adaptive for uneven lighting)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

    # 6. De-skew (rotation correction)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = thresh.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    deskewed = cv2.warpAffine(thresh, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    # 7. Noise removal (median blur)
    denoised = cv2.medianBlur(deskewed, 3)

    # 8. Sharpening (helps with faint text)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # 9. Morphological ops to connect broken characters
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.dilate(sharpened, kernel, iterations=1)

    return processed

def extract_aadhaar_data(image_path):
    try:
        img = cv2.imread(image_path)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
        
        pil_img = Image.fromarray(enhanced)
        
        configs = [
            r'--oem 3 --psm 6',      # Default
            r'--oem 3 --psm 3',      # Fully automatic page segmentation
            r'--oem 3 --psm 4',      # Single column text
            r'--oem 3 --psm 11',     # Sparse text
        ]
        
        best_text = ""
        for config in configs:
            try:
                text = pytesseract.image_to_string(pil_img, lang='eng+hin', config=config)
                if len(text.strip()) > len(best_text.strip()):
                    best_text = text
            except:
                continue
        
        if not best_text.strip():
            # Fallback: trying with original image
            original_pil = Image.open(image_path)
            best_text = pytesseract.image_to_string(original_pil, lang='eng', config=r'--oem 3 --psm 6')
        
        # Step 3: Parsing the text with improved logic
        lines = [line.strip() for line in best_text.split('\n') if line.strip()]
        
        aadhaar_data = {
            "Name": None,
            "Date of Birth": None,
            "Gender": None,
            "Aadhaar Number": None,
            "raw_text": best_text
        }
        
        full_text = ' '.join(lines)
        
        aadhaar_patterns = [
            r'\b\d{4}\s+\d{4}\s+\d{4}\b',      # Standard format with spaces
            r'\b\d{4}-\d{4}-\d{4}\b',          # With hyphens
            r'\b\d{12}\b',                     # Without separators
        ]
        
        for pattern in aadhaar_patterns:
            match = re.search(pattern, full_text)
            if match:
                number = re.sub(r'\D', '', match.group(0))  
                if len(number) == 12:
                    aadhaar_data["Aadhaar Number"] = f"{number[:4]} {number[4:8]} {number[8:12]}"
                    break
        
        dob_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',      # DD/MM/YYYY or DD-MM-YYYY
            r'\b\d{1,2}\s+\d{1,2}\s+\d{4}\b',        # DD MM YYYY
            r'DOB[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{4}', # DOB: DD/MM/YYYY
        ]
        
        for pattern in dob_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                dob_text = match.group(0)
                date_match = re.search(r'\d{1,2}[/\-\s]\d{1,2}[/\-\s]\d{4}', dob_text)
                if date_match:
                    aadhaar_data["Date of Birth"] = date_match.group(0).replace(' ', '/')
                    break
        
        gender_patterns = [
            r'\b(male|female|पुरुष|महिला)\b',
            r'\b(M|F|पु|म)\b',
        ]
        
        for pattern in gender_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                gender_text = match.group(0).lower()
                if gender_text in ['male', 'm', 'पुरुष', 'पु']:
                    aadhaar_data["Gender"] = "Male"
                elif gender_text in ['female', 'f', 'महिला', 'म']:
                    aadhaar_data["Gender"] = "Female"
                break
        
        def is_likely_name(text):
            if len(text) < 2 or len(text) > 50:
                return False
            if re.search(r'\d', text):  
                return False
            if any(word.lower() in text.lower() for word in ['government', 'india', 'aadhaar', 'card', 'dob', 'address']):
                return False
            if re.match(r'^[A-Za-z\s\.]+$', text): 
                return True
            return False
        
        name_candidates = []
        
        for line in lines:
            if is_likely_name(line):
                name_candidates.append(line)
        
        name_patterns = [
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',  # Proper case names
            r'(?:श्री|श्रीमती|Mr|Ms|Miss)\s+([A-Za-z\s]+)',        # With titles
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, full_text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                if is_likely_name(match.strip()):
                    name_candidates.append(match.strip())
        
        if name_candidates:
            best_name = max(name_candidates, key=len)
            aadhaar_data["Name"] = best_name
        
        return aadhaar_data
        
    except Exception as e:
        return {"error": str(e)}


def preprocess_aadhaar_image_simple(image_path):
    #Simplified preprocessing
    img = cv2.imread(image_path)
    
    # Converting to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Simple contrast enhancement
    enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
    
    # slight blur to reduce noise
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return denoised


# Alternative function
def extract_aadhaar_data_alternative(image_path):
    """Alternative approach using different OCR strategies"""
    try:
        original_img = Image.open(image_path)
    
        texts = []
        
        # English only
        try:
            text1 = pytesseract.image_to_string(original_img, lang='eng', config=r'--oem 3 --psm 6')
            texts.append(text1)
        except:
            pass
        
        # Grayscale conversion
        try:
            img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img_pil = Image.fromarray(img_cv)
            text2 = pytesseract.image_to_string(img_pil, lang='eng+hin', config=r'--oem 3 --psm 3')
            texts.append(text2)
        except:
            pass
        
        # Threshold processing
        try:
            img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            _, thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_pil = Image.fromarray(thresh)
            text3 = pytesseract.image_to_string(img_pil, lang='eng', config=r'--oem 3 --psm 11')
            texts.append(text3)
        except:
            pass
        
        combined_text = '\n'.join(texts)
        
        aadhaar_data = {
            "Name": None,
            "Date of Birth": None,
            "Gender": None,
            "Aadhaar Number": None,
            "raw_text": combined_text
        }
        
        full_text = combined_text
        
        # Extracting Aadhaar number
        aadhaar_match = re.search(r'\b\d{4}\s*\d{4}\s*\d{4}\b', full_text)
        if aadhaar_match:
            number = re.sub(r'\D', '', aadhaar_match.group(0))
            if len(number) == 12:
                aadhaar_data["Aadhaar Number"] = f"{number[:4]} {number[4:8]} {number[8:12]}"
        
        dob_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',                   # DD/MM/YYYY
            r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{4}\b',           # DD-MM-YYYY  
            r'\b\d{1,2}\s+\d{1,2}\s+\d{4}\b',               # DD MM YYYY
            r'DOB:\s*\d{1,2}/\d{1,2}/\d{4}',                # DOB: DD/MM/YYYY
            r'DOB[:\s]*\d{1,2}[/\-]\d{1,2}[/\-]\d{4}',      # DOB: DD-MM-YYYY
        ]
        
        for pattern in dob_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                # Extract the date portion
                date_part = re.search(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{4}', match.group(0))
                if date_part:
                    aadhaar_data["Date of Birth"] = date_part.group(0).replace('-', '/')
                    break
        
        dob_match = re.search(r'\d{1,2}[/\-\s]\d{1,2}[/\-\s]\d{4}', full_text)
        if dob_match:
            aadhaar_data["Date of Birth"] = dob_match.group(0)
        
        if re.search(r'\b(male|पुरुष)\b', full_text, re.IGNORECASE):
            aadhaar_data["Gender"] = "Male"
        elif re.search(r'\b(female|महिला)\b', full_text, re.IGNORECASE):
            aadhaar_data["Gender"] = "Female"
        
        lines = [line.strip() for line in combined_text.split('\n') if line.strip()]
        for i, line in enumerate(lines):
            if 'DOB' in line or re.search(r'\d{2}/\d{2}/\d{4}', line):
                # Look for name in previous lines
                for j in range(max(0, i-3), i):
                    if re.match(r'^[A-Za-z\s]+$', lines[j]) and len(lines[j]) > 3:
                        aadhaar_data["Name"] = lines[j]
                        break
                break
        
        return aadhaar_data
        
    except Exception as e:
        return {"error": str(e)}

def validate_aadhaar_number(aadhaar_data):
    """Validate Aadhaar number using Verhoeff checksum"""
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
    """Complete fraud analysis of the image"""
    results = {
        "exif_analysis": get_exif_data(image_path),
        "object_detection": detect_objects_yolo(image_path),
        "ocr_analysis": extract_aadhaar_data(image_path),
        "fraud_indicators": []
    }
    
   
    if "error" not in results["ocr_analysis"]:
        results["aadhaar_validation"] = validate_aadhaar_number(results["ocr_analysis"])
    
    fraud_score = 0
    
    if not results["exif_analysis"] or len(results["exif_analysis"]) == 0:
        results["fraud_indicators"].append("No EXIF metadata found (possible digital manipulation)")
        fraud_score += 1
    
    if "error" not in results["object_detection"]:
        if results["object_detection"].get("fraud_indicator"):
            results["fraud_indicators"].append("Non-human objects detected in image")
            fraud_score += 2
    
    if "aadhaar_validation" in results:
        if not results["aadhaar_validation"]["valid"]:
            results["fraud_indicators"].append(f"Invalid Aadhaar number: {results['aadhaar_validation']['reason']}")
            fraud_score += 2
    
    # Overall assessment
    if fraud_score >= 3:
        results["assessment"] = "HIGH FRAUD RISK"
    elif fraud_score >= 1:
        results["assessment"] = "MODERATE FRAUD RISK"
    else:
        results["assessment"] = "LOW FRAUD RISK"
    
    results["fraud_score"] = fraud_score
    
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
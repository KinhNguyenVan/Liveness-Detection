import cv2
import torch
import numpy as np
from model import LivenessModel, preprocessor


face_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
face_model.classes = [0]  


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    checkpoint = torch.load("D:/CAKE/checkpoint/model_dino_lora.pt", 
                            map_location=torch.device(device))
    model = LivenessModel(checkpoint['args'])
    processor = preprocessor(model.args)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  
    print("Liveness model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)


print("Processor structure:", dir(processor))


def preprocess_face(face_img):
    try:
      
        if face_img.shape[0] < 10 or face_img.shape[1] < 10:
            print("Face too small, dimensions:", face_img.shape)
            return None
            
   
        print(f"Face dimensions: {face_img.shape}")
        
       
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
        processed = processor(face_rgb)
        print("Processor output keys:", processed.keys())
        
       
        if "pixel_values" not in processed:
            raise KeyError("pixel_values not found in processor output")
            
        pixel_values = processed["pixel_values"]
        print(f"Pixel values shape: {pixel_values.shape}")
        
        # Nếu pixel_values có batch dimension, giữ nguyên
        if len(pixel_values.shape) == 4:  # [batch, channels, height, width]
            return pixel_values.to(device)
        else:
            
            return pixel_values.unsqueeze(0).to(device)
                
            
    except Exception as e:
        print(f"Error in preprocess_face: {e}")
        return None


def classify_face(face_img):
    try:
        face_tensor = preprocess_face(face_img)
        if face_tensor is None:
            print("Face tensor is None, returning unknown")
            return "unknown", 0.0
            
        idx2label = {0: "spoof", 1: "normal"}
        
       
        print(f"Input tensor shape: {face_tensor.shape}")
        
        with torch.no_grad():
            try:
                output = model(face_tensor)
                print(f"Model output: {output}, shape: {output.shape}")
                
             
                if isinstance(output, dict) and "logits" in output:
                    scores = torch.nn.functional.softmax(output["logits"], dim=-1)
                else:
                    scores = torch.nn.functional.softmax(output, dim=-1)
                
                print(f"Softmax scores: {scores}")
          
                predicted_class = torch.argmax(scores, dim=-1).item()
                confidence = scores[0][predicted_class].item()
                
                print(f"Predicted class: {predicted_class}, confidence: {confidence}")
                return idx2label[predicted_class], confidence
                
            except Exception as e:
                print(f"Error in model inference: {e}")
                return "error", 0.0
    except Exception as e:
        print(f"Error in classify_face: {e}")
        return "error", 0.0

# Khởi động webcam
cap = cv2.VideoCapture(0)

# Kiểm tra xem webcam có mở được không
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        break
    
    # Tạo bản sao của frame để hiển thị
    display_frame = frame.copy()
    
    # Phát hiện khuôn mặt sử dụng YOLOv5
    results = face_model(frame)
    
  
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Xử lý từng khuôn mặt được phát hiện
    detections = results.xyxy[0].cpu().numpy() if len(results.xyxy) > 0 else []
    
    for detection in detections:
        # Lấy tọa độ bounding box
        x1, y1, x2, y2, conf, class_id = detection
        
        # Chỉ xử lý nếu là người (class 0) và độ tin cậy > 0.5
        if int(class_id) == 0 and conf > 0.5:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Trích xuất vùng khuôn mặt
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size > 0:
            
                classification, confidence = classify_face(face_roi)
                
                # Vẽ bounding box và nhãn phân loại
                color = (0, 255, 0) if classification == 'normal' else (0, 0, 255)
                if classification == "unknown" or classification == "error":
                    color = (255, 0, 255)  # Màu tím cho unknown/error
                
                label = f"{classification} ({confidence:.2f})"
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Thêm background cho text để dễ đọc
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(display_frame, (x1, y1 - text_size[1] - 10), 
                             (x1 + text_size[0], y1), color, -1)
                cv2.putText(display_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
   
    cv2.imshow("Face Liveness Detection", display_frame)
    
    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
# test.py
import torch
import cv2
import numpy as np
from pathlib import Path
from config import CONFIG
from model import build_model
from utils import non_max_suppression, scale_coords, plot_one_box
from data import letterbox
import os

class JellyfishDetector:
    # JellyfishDetector 클래스 초기화 시 임계값을 더 엄격하게 설정
    def __init__(self, weights_path, conf_thres=0.5, iou_thres=0.45):  # conf_thres를 0.25에서 0.5로 높임
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 클래스 이름 설정
        self.names = [
            'barrel_jellyfish',
            'blue_jellyfish',
            'compass_jellyfish',
            'lions_mane_jellyfish',
            'mauve_stinger_jellyfish',
            'moon_jellyfish'
        ]
        
        # 모델 로드
        self.model = build_model()
        try:
            # .pth 파일 로드 시도
            checkpoint = torch.load(weights_path, map_location=self.device)
            # checkpoint에서 model_state_dict 추출
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def preprocess_image(self, img_path):
        img0 = cv2.imread(str(img_path))
        if img0 is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        img = letterbox(img0, new_shape=640)[0]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img, img0

    def detect(self, img_path):
        img, img0 = self.preprocess_image(img_path)
        
        with torch.no_grad():
            pred = self.model(img)
            print("Model output shape:", [p.shape for p in pred])
            
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            
            pred[..., 4:] = torch.sigmoid(pred[..., 4:])
            
            # max_det 매개변수 제거
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        results = []
        class_counts = {}  # 클래스별 카운트 추적
        
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
                # confidence 기준으로 정렬하고 상위 10개만 선택
                det = det[det[:, 4].argsort(descending=True)]
                det = det[:10]  # 최대 10개로 제한
                
                for *xyxy, conf, cls in det:
                    cls_idx = int(cls)
                    if cls_idx < 0 or cls_idx >= len(self.names):
                        continue
                    
                    # 클래스별 최대 2개까지만 검출
                    class_name = self.names[cls_idx]
                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                    if class_counts[class_name] >= 2:
                        continue
                    class_counts[class_name] += 1
                    
                    conf = torch.clamp(conf, 0, 1)
                    
                    label = f'{class_name} {conf:.2f}'
                    plot_one_box(xyxy, img0, label=label, color=self.colors[cls_idx])
                    
                    results.append({
                        'bbox': [coord.item() for coord in xyxy],
                        'confidence': conf.item(),
                        'class': class_name
                    })

        return img0, results

def test_single_image(detector, image_path, save_dir):
    """단일 이미지 테스트"""
    try:
        result_img, detections = detector.detect(image_path)
        
        # 결과 출력
        print(f"\nResults for {image_path}:")
        for det in detections:
            print(f"Class: {det['class']}, Confidence: {det['confidence']:.2f}")
        
        # 결과 이미지 저장
        save_path = os.path.join(save_dir, f"result_{os.path.basename(image_path)}")
        cv2.imwrite(save_path, result_img)
        print(f"Result saved to: {save_path}")
        
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def test_directory(detector, test_dir, save_dir):
    """테스트 디렉토리의 모든 이미지 테스트"""
    test_dir = Path(test_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 지원하는 이미지 확장자
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    successful = 0
    failed = 0
    
    for class_dir in test_dir.iterdir():
        if class_dir.is_dir():
            print(f"\nProcessing class: {class_dir.name}")
            
            # 클래스별 결과 저장 디렉토리 생성
            class_save_dir = save_dir / class_dir.name
            class_save_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in img_extensions:
                    if test_single_image(detector, img_path, class_save_dir):
                        successful += 1
                    else:
                        failed += 1
    
    print(f"\nTest completed: {successful} successful, {failed} failed")

def main():
    # 경로 설정
    base_dir = Path(os.getcwd())
    weights_path = base_dir / 'runs' / 'train' / 'exp' / 'best_model_20250117_221328.pth'  # 수정된 부분
    test_dir = base_dir / 'jellyfish' / 'Train_Test_Valid' / 'test'
    save_dir = base_dir / 'results'
    
    # 설정 출력
    print("Initializing test with following settings:")
    print(f"Weights path: {weights_path}")
    print(f"Test directory: {test_dir}")
    print(f"Save directory: {save_dir}")
    
    # 가중치 파일 존재 확인
    if not weights_path.exists():
        print(f"Error: Weights file not found at {weights_path}")
        print("\nAvailable weight files in runs/train/exp/:")
        weight_dir = base_dir / 'runs' / 'train' / 'exp'
        for file in weight_dir.glob('*.pth'):
            print(f"- {file.name}")
        
        # 사용자에게 가중치 파일 선택 요청
        print("\nPlease enter the name of the weight file you want to use:")
        weight_name = input("> ")
        weights_path = weight_dir / weight_name
        
        if not weights_path.exists():
            print(f"Error: Selected weight file does not exist: {weights_path}")
            return
    
    # 디텍터 초기화
    detector = JellyfishDetector(weights_path)
    
    # 테스트 모드 선택
    while True:
        print("\nSelect test mode:")
        print("1. Test single image")
        print("2. Test entire test directory")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            img_path = input("Enter the path to the image: ")
            test_single_image(detector, Path(img_path), save_dir)
        elif choice == '2':
            test_directory(detector, test_dir, save_dir)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
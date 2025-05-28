import sys
import os
import numpy as np
from models.food_classifier import FoodClassifier
from utils.image_processor import ImageProcessor

# 경로 설정
MODEL_PATH = "models/saved_model_final.keras"
CLASS_MAPPING_PATH = "models/class_mapping.txt"

def load_class_mapping(path):
    mapping = {}
    try:
        with open(path, "r", encoding="cp949") as f:
            for line in f:
                if ":" in line:
                    name, idx = line.strip().split(":")
                    mapping[int(idx)] = name
    except UnicodeDecodeError:
        # cp949 실패 시 utf-8로 다시 시도
        print("Warning: cp949 decode failed, trying utf-8")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    name, idx = line.strip().split(":")
                    mapping[int(idx)] = name
    except Exception as e:
        print(f"Error loading class mapping: {e}")
        sys.exit(1)
    return mapping

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python predict_image.py [이미지 파일 또는 폴더 경로]")
        sys.exit(1)

    input_path = sys.argv[1]

    class_mapping = load_class_mapping(CLASS_MAPPING_PATH)
    num_classes = len(class_mapping)
    classifier = FoodClassifier(num_classes=num_classes)
    classifier.load_model(MODEL_PATH)
    processor = ImageProcessor(target_size=(224, 224))

    # 입력 경로가 폴더인지 파일인지 확인
    if os.path.isdir(input_path):
        image_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        if not image_files:
            print(f"폴더에 이미지 파일이 없습니다: {input_path}")
            sys.exit(1)

        print(f"'{input_path}' 폴더 내 이미지 예측 시작...")
        for image_path in image_files:
            print(f"Processing {image_path}...")
            img = processor.preprocess_image(image_path)
            if img is not None:
                preds = classifier.predict(img)
                pred_idx = np.argmax(preds)
                pred_class = class_mapping.get(pred_idx, "알 수 없음")
                print(f"  예측 결과: {pred_class}")
            else:
                print(f"  이미지 전처리 실패: {image_path}")
    elif os.path.isfile(input_path):
        print(f"Processing {input_path}...")
        img = processor.preprocess_image(input_path)
        if img is not None:
            preds = classifier.predict(img)
            pred_idx = np.argmax(preds)
            pred_class = class_mapping.get(pred_idx, "알 수 없음")
            print(f"예측 결과: {pred_class}")
        else:
            print("이미지 전처리에 실패했습니다.")
    else:
        print(f"잘못된 경로입니다: {input_path}. 이미지 파일 또는 폴더 경로를 입력해주세요.")
        sys.exit(1) 
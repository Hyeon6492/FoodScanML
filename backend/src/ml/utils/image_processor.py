import cv2
import numpy as np
from PIL import Image
import os

class ImageProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def preprocess_image(self, image_path):
        """
        이미지를 전처리하여 모델 입력에 맞는 형태로 변환합니다.
        """
        try:
            # 이미지 로드
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")

            # BGR에서 RGB로 변환
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 이미지 크기 조정
            img = cv2.resize(img, self.target_size)

            # 정규화 (0-1 범위로)
            img = img.astype(np.float32) / 255.0

            # 배치 차원 추가
            img = np.expand_dims(img, axis=0)

            return img
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None

    def load_and_preprocess_batch(self, image_paths):
        """
        여러 이미지를 배치로 전처리합니다.
        """
        processed_images = []
        for path in image_paths:
            processed = self.preprocess_image(path)
            if processed is not None:
                processed_images.append(processed)
        
        if processed_images:
            return np.vstack(processed_images)
        return None

    def save_processed_image(self, image, output_path):
        """
        전처리된 이미지를 저장합니다.
        """
        try:
            # 배치 차원 제거
            if len(image.shape) == 4:
                image = image[0]

            # 0-1 범위에서 0-255 범위로 변환
            image = (image * 255).astype(np.uint8)

            # RGB에서 BGR로 변환
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 이미지 저장
            cv2.imwrite(output_path, image)
            return True
        except Exception as e:
            print(f"Error saving processed image: {str(e)}")
            return False 
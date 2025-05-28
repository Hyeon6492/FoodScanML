import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .image_processor import ImageProcessor

class DataLoader:
    def __init__(self, data_dir, target_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.target_size = target_size
        self.batch_size = batch_size
        self.image_processor = ImageProcessor(target_size)
        self.class_names = self._get_class_names()

    def _get_class_names(self):
        """
        데이터 디렉토리에서 클래스 이름을 가져옵니다.
        """
        return sorted([d for d in os.listdir(self.data_dir) 
                      if os.path.isdir(os.path.join(self.data_dir, d))])

    def create_data_generators(self, validation_split=0.2):
        """
        학습 및 검증 데이터 생성기를 생성합니다.
        """
        # 데이터 증강 설정
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )

        # 검증 데이터 생성기 (증강 없음)
        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )

        # 학습 데이터 생성기
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )

        # 검증 데이터 생성기
        validation_generator = validation_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )

        return train_generator, validation_generator

    def load_single_image(self, image_path):
        """
        단일 이미지를 로드하고 전처리합니다.
        """
        return self.image_processor.preprocess_image(image_path)

    def get_class_mapping(self):
        """
        클래스 이름과 인덱스 매핑을 반환합니다.
        """
        return {class_name: idx for idx, class_name in enumerate(self.class_names)}

    def get_class_name(self, class_idx):
        """
        클래스 인덱스에 해당하는 클래스 이름을 반환합니다.
        """
        return self.class_names[class_idx] if 0 <= class_idx < len(self.class_names) else None 
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

class FoodClassifier:
    def __init__(self, num_classes, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        """
        MobileNetV2 기반의 전이학습 모델을 구축합니다.
        """
        # MobileNetV2 기본 모델 로드
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )

        # 기본 모델 동결
        base_model.trainable = False

        # 모델 구성
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # 모델 컴파일
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, train_data, validation_data, epochs=10, batch_size=32):
        """
        모델을 학습시킵니다.
        """
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )
        return history

    def predict(self, image):
        """
        이미지에 대한 예측을 수행합니다.
        """
        predictions = self.model.predict(image)
        return predictions

    def save_model(self, model_path):
        """
        모델을 저장합니다.
        """
        self.model.save(model_path)

    def load_model(self, model_path):
        """
        저장된 모델을 로드합니다.
        """
        if os.path.exists(model_path):
            self.model = models.load_model(model_path)
            return True
        return False

    def fine_tune(self, train_data, validation_data, epochs=5, batch_size=32):
        """
        모델을 미세 조정합니다.
        """
        # 기본 모델의 일부 레이어 동결 해제
        for layer in self.model.layers[0].layers[-20:]:
            layer.trainable = True

        # 모델 재컴파일
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # 미세 조정 학습
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )
        return history 
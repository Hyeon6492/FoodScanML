import os
import argparse
from models.food_classifier import FoodClassifier
from utils.data_loader import DataLoader

def train_model(data_dir, model_save_path, epochs=10, batch_size=32):
    """
    모델을 학습시키는 함수
    """
    # 데이터 로더 초기화
    data_loader = DataLoader(data_dir, batch_size=batch_size)
    train_generator, validation_generator = data_loader.create_data_generators()

    # 클래스 수 가져오기
    num_classes = len(data_loader.class_names)

    # 모델 초기화
    model = FoodClassifier(num_classes=num_classes)

    # 모델 학습
    history = model.train(
        train_generator,
        validation_generator,
        epochs=epochs,
        batch_size=batch_size
    )

    # 모델 미세 조정
    history_fine = model.fine_tune(
        train_generator,
        validation_generator,
        epochs=5,
        batch_size=batch_size
    )

    # 모델 저장
    model.save_model(model_save_path)

    # 클래스 매핑 저장
    class_mapping = data_loader.get_class_mapping()
    with open(os.path.join(os.path.dirname(model_save_path), 'class_mapping.txt'), 'w') as f:
        for class_name, idx in class_mapping.items():
            f.write(f"{class_name}:{idx}\n")

    return history, history_fine

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train food classification model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')

    args = parser.parse_args()

    # 모델 학습
    history, history_fine = train_model(
        args.data_dir,
        args.model_save_path,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    print("Training completed!")
    print(f"Model saved to: {args.model_save_path}") 
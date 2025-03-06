import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')


def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()


def extract_features_for_dataset(images_dir, output_file='data/processed/features.pkl'):
    features_dict = {}
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    image_files = os.listdir(images_dir)
    total_images = len(image_files)
    print(f"Знайдено {total_images} зображень.")

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        features = extract_features(img_path, model)
        features_dict[img_file] = features
        if (idx + 1) % 100 == 0:
            print(f"Оброблено {idx + 1} з {total_images} зображень.")

    with open(output_file, 'wb') as f:
        pickle.dump(features_dict, f)
    print(f"Ознаки збережено в файлі: {output_file}")
    return features_dict


if __name__ == '__main__':
    images_dir = 'data/raw/Flickr8k/images'
    features = extract_features_for_dataset(images_dir)

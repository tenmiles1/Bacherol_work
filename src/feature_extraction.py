import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

model = InceptionV3(weights='imagenet', include_top=False)
model.trainable = False

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0)
    return np.reshape(features, (64, 2048))

def extract_features_for_split(images_dir, split_file, output_file):
    features_dict = {}
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(split_file, "r") as f:
        image_ids = [line.strip() for line in f.readlines()]
    image_files = [img_id if img_id.lower().endswith(".jpg") else img_id + ".jpg" for img_id in image_ids]

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        if not os.path.exists(img_path):
            print(f"не знайдено: {img_file} ")
            continue
        try:
            features = extract_features(img_path)
            image_id = img_file.split(".jpg")[0]
            features_dict[image_id] = features
            if (idx + 1) % 50 == 0:
                print(f"Оброблено {idx + 1}/{len(image_files)}")
        except Exception as e:
            print(f"Помилка при {img_file}: {e}")

    with open(output_file, 'wb') as f:
        pickle.dump(features_dict, f)

    print(f"Збережено {len(features_dict)} features в {output_file}")

if __name__ == '__main__':
    images_dir = 'data/raw/Flickr30k/Flickr30k/flickr30k_images'
    split_file = 'data/splits/Flickr_30k.testImages.txt'
    output_file = 'data/processed/features_test.pkl'
    extract_features_for_split(images_dir, split_file, output_file)

import os
import random
import shutil


script_dir = os.path.dirname(os.path.abspath(__file__))
imgs_path = os.path.join(script_dir, '..', 'imgs')
train_path = os.path.join(imgs_path, 'Train')
test_path = os.path.join(imgs_path, 'Test')


TEST_IMAGES_PER_CLASS = 10
SEED = 42


random.seed(SEED)
os.makedirs(test_path, exist_ok=True)


print("\nCriando split de teste...")
print(f"Train: {train_path}")
print(f"Test:  {test_path}\n")


for class_folder in sorted(os.listdir(train_path)):
    train_class_path = os.path.join(train_path, class_folder)
    if not os.path.isdir(train_class_path):
        continue

    test_class_path = os.path.join(test_path, class_folder)
    os.makedirs(test_class_path, exist_ok=True)

    test_image_names = []
    for img_name in os.listdir(test_class_path):
        if img_name.lower().endswith('.jpg'):
            test_image_names.append(img_name)

    images_to_move = TEST_IMAGES_PER_CLASS - len(test_image_names)

    if images_to_move <= 0:
        print(f"{class_folder}: teste ja tem {len(test_image_names)} imagens")
        continue

    image_names = []
    for img_name in os.listdir(train_class_path):
        if img_name.lower().endswith('.jpg'):
            image_names.append(img_name)

    image_names = sorted(image_names)
    random.shuffle(image_names)

    selected_images = image_names[:images_to_move]
    moved = 0

    for img_name in selected_images:
        source = os.path.join(train_class_path, img_name)
        destination = os.path.join(test_class_path, img_name)

        if os.path.exists(destination):
            continue

        shutil.move(source, destination)
        moved += 1

    print(f"{class_folder}: {moved} imagens movidas para teste")


print("\nSplit finalizado!")

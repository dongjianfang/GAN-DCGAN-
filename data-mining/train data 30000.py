import os
import random
import shutil

def random_select_images(dataset_path, num_images, save_path):
    image_list = []

    # 遍历数据集目录并获取所有图像文件的路径
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                image_list.append(image_path)

    # 从图像列表中随机选择指定数量的图像
    selected_images = random.sample(image_list, num_images)

    # 将选中的图像复制到保存路径中
    for i, image_path in enumerate(selected_images):
        dst_path = os.path.join(save_path, f"image_{i+1}.jpg")
        shutil.copyfile(image_path, dst_path)

    print(f"{num_images} images have been saved to {save_path}")

# 调用函数，传入自定义数据集的路径、要选择的图像数量和保存路径
random_select_images("E:/GAN1/face/img_align_celeba", 30000, "E:/GAN1/imaged2/real_image")
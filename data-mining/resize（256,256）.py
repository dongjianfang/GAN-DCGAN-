import os
from PIL import Image

# 原始图像文件夹路径
folder_path = 'E:/GAN1/imaged/generator_image2'

# 调整后的图像尺寸
new_width = 256
new_height = 256

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件扩展名，确保只处理图像文件
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # 构建完整的图像文件路径
        image_path = os.path.join(folder_path, filename)

        # 打开图像
        image = Image.open(image_path)

        # 调整图像尺寸
        new_image = image.resize((new_width, new_height))

        # 将重塑后的图像保存到原始路径
        new_image.save(image_path)
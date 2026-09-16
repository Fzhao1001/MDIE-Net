from PIL import Image

import os


def convert_images_to_png(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):  # 添加更多你想要的格式

            with Image.open(os.path.join(input_dir, filename)) as img:
                rgb_img = img.convert('RGB')  # 如果需要，可以转换为RGB格式

                rgb_img.save(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png"), "PNG")

            # 使用函数


input_directory = ''

output_directory = ''

# convert_images_to_png(input_directory, output_directory)
rgb_img = Image.open(os.path.join(input_directory, "303.jpg")).convert('RGB')
rgb_img.save(os.path.join(output_directory , "303.png"), "PNG")
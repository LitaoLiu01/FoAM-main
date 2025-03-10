from PIL import Image
import os

def resize_image(input_path, output_path, size=(512, 512)):
    """
    将图片调整为指定大小。

    :param input_path: 输入图片的路径
    :param output_path: 输出图片的路径
    :param size: 目标大小，默认为 (512, 512)
    """
    if not os.path.exists(input_path):
        print(f"输入文件路径不存在: {input_path}")
        return

    with Image.open(input_path) as img:
        resized_img = img.resize(size, Image.Resampling.LANCZOS)
        resized_img.save(output_path)
        print(f"图片已保存为 {output_path}")

# 示例使用
input_image_path = '/home/liulitao/CASIA_Intern/BaseLines/DataCaptuer/FinalDataCap/SimpleLevel/PutStuff2Pan/model/image/texture_cabinet_3.jpg'
output_image_path = '/home/liulitao/CASIA_Intern/BaseLines/DataCaptuer/FinalDataCap/SimpleLevel/PutStuff2Pan/model/image/texture_cabinet_3_resize.png'
resize_image(input_image_path, output_image_path)

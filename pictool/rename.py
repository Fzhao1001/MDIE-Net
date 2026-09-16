import os
import uuid


def batch_rename_images(folder_path, sort_method='name'):
    """
    将文件夹内的图片文件按顺序重命名为1/2/3...格式
    :param folder_path: 目标文件夹路径
    :param sort_method: 排序方式 ['name'文件名|'modified'修改时间|'created'创建时间]
    """
    # 支持的图片格式（可自行扩展）
    image_exts = ('.png', '.jpg', '.jpeg', '.gif',
                  '.bmp', '.tiff', '.webp', '.svg')

    # 验证文件夹有效性
    if not os.path.isdir(folder_path):
        raise ValueError("无效的文件夹路径")

    # 获取文件列表并过滤
    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(image_exts) and os.path.isfile(os.path.join(folder_path, f))]

    # 设置排序方式
    if sort_method == 'name':
        files.sort()
    elif sort_method == 'modified':
        files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
    elif sort_method == 'created':
        files.sort(key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    else:
        raise ValueError("不支持的排序方式，可选：name/modified/created")

    # 双重保险重命名流程
    temp_files = []
    try:
        # 第一阶段：重命名为临时文件
        for file in files:
            src = os.path.join(folder_path, file)
            ext = os.path.splitext(file)[1].lower()  # 统一小写扩展名
            temp_name = f"temp_{uuid.uuid4().hex}{ext}"
            dst = os.path.join(folder_path, temp_name)
            os.rename(src, dst)
            temp_files.append((dst, ext))
        new_dir = ""
        # 第二阶段：正式重命名
        for index, (temp_path, ext) in enumerate(temp_files, 1):
            new_name = f"{index}{ext}"
            new_path = os.path.join(folder_path, new_name)
            os.rename(temp_path, new_path)
            print(f"成功：{os.path.basename(temp_path)} -> {new_name}")

    except Exception as e:
        print(f"操作中断，正在回滚...（错误信息：{str(e)}）")
        # 错误回滚机制
        for temp_path, _ in temp_files:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        raise


if __name__ == "__main__":
    # 用户交互
    target_folder = ""

    sort_mapping = {
        '1': 'name',
        '2': 'modified',
        '3': 'created'
    }

    batch_rename_images(
        folder_path=target_folder,
        sort_method="name"
    )
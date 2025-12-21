import os

file_path = 'scene/neural_3D_dataset_NDC.py'

with open(file_path, 'r') as f:
    content = f.read()

search_str = """        img = img.resize(self.img_wh, Image.LANCZOS)"""

replace_str = """        img = img.resize(self.img_wh, Image.BILINEAR)"""

if search_str in content:
    content = content.replace(search_str, replace_str)
    print("Applied resize optimization to scene/neural_3D_dataset_NDC.py")
else:
    print("Could not find context for resize optimization in scene/neural_3D_dataset_NDC.py")

with open(file_path, 'w') as f:
    f.write(content)

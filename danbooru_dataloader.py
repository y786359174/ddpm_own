import json
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

def load_metadata(json_file):
    with open(json_file, 'r') as f:
        metadata = json.load(f)
    
    # 创建一个字典，键为图像ID，值为标签的列表
    metadata_dict = {}
    for entry in metadata:
        image_id = entry['id']
        tags = [tag['name'] for tag in entry['tags']]
        metadata_dict[image_id] = tags
    
    return metadata_dict


class CustomDataset(Dataset):
    def __init__(self, image_root, metadata_dict, transform=None):
        self.image_root = image_root
        self.metadata_dict = metadata_dict
        self.image_ids = list(metadata_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        folder_num = int(image_id) % 150
        folder_name = f"{folder_num:04d}"
        image_file = f"{image_id}.jpg"
        image_path = os.path.join(self.image_root, folder_name, image_file)
        
        image = Image.open(image_path).convert('RGB')
        
        tags = self.metadata_dict[image_id]
        
        if self.transform:
            image = self.transform(image)
        
        return image, tags
    


if __name__ == '__main__':

    # 假设图像存储在 'images/' 目录下
    image_root = 'images/'
    # 假设元数据文件为 'metadata.json'
    metadata_file = 'metadata.json'

    # 加载元数据
    metadata_dict = load_metadata(metadata_file)

    # 定义图像的预处理操作
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # 创建自定义数据集
    dataset = CustomDataset(image_root=image_root, metadata_dict=metadata_dict, transform=transform)

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)


    for images, tags in dataloader:

        # 在这里处理图像和标签
        print(tags)
        
        pass
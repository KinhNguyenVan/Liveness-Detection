from torch.utils.data import DataLoader, Dataset
import os  
from PIL import Image
from transformers import AutoImageProcessor
from torchvision import transforms
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, modelname: str,dev = False):
        self.root_dir = root_dir
        self.processor = AutoImageProcessor.from_pretrained(modelname,use_fast = True)
        self.dev = dev
        self.image_infos = []  
        for label in ["normal", "spoof"]:
            label_dir = os.path.join(root_dir, label)
            if os.path.exists(label_dir):
                for file_name in os.listdir(label_dir):
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        file_path = os.path.join(label_dir, file_name)
                        label_id = 0 if label == "spoof" else 1
                        if self.dev:
                            self.image_infos.append((file_path, label_id, 0))  
                        else:
                            
                            for angle in [0, 30, 60, 90]:
                                self.image_infos.append((file_path, label_id, angle))
    def __len__(self):
        return len(self.image_infos)

    def __getitem__(self, idx):
        img_path, label, angle = self.image_infos[idx]
        image = Image.open(img_path).convert("RGB")
        if angle != 0 and self.dev is False:
            image = transforms.functional.rotate(image, angle)

        image = self.processor(image, return_tensors="pt")['pixel_values'].squeeze(0)
        return image, label
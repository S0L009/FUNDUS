from torch.utils.data import Dataset
from PIL import Image
import os

class Segm_ds(Dataset):
    def __init__(self, tt, trf, mask_trf) -> None:
        super().__init__()
        if tt == 'train':
            self.img_path = r"/mnt/c/Users/Krish/Downloads/jet/Ds/augm/segm_train/img"
            self.mask_path = r"/mnt/c/Users/Krish/Downloads/jet/Ds/augm/segm_train/mask"
        else:
            self.img_path = r"/mnt/c/Users/Krish/Downloads/jet/Ds/augm/segm_test/img"
            self.mask_path = r"/mnt/c/Users/Krish/Downloads/jet/Ds/augm/segm_test/mask"

        self.paths = sorted(os.listdir(self.img_path))
        self.trf = trf
        self.mask_trf = mask_trf

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_path + '/' + self.paths[idx]).convert('RGB')
        mask = Image.open(self.mask_path + '/' + self.paths[idx]).convert('L')
        img = self.trf(img)
        mask = self.mask_trf(mask)
        return img, mask
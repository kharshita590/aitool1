import numpy as np
import os
# import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)

        files = os.listdir(root_dir)

        self.data = list(zip(files, list(range(len(files)))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir)
        img_size = (256, 256)
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        image = np.array(Image.open(os.path.join(
            root_and_dir, img_file)))
        pil_image = Image.fromarray(np.uint8(image))
        image_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(0.3),
            transforms.ToTensor(),
            normalize,
        ])
        # img_s = "/home/akhilesh/Desktop/dataset/images2.png"
        image = image_transform(pil_image)
        return image


def test():
    dataset = MyImageFolder(root_dir="/home/akhilesh/Desktop/dataset")
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    for image in loader:
        print(image.shape)

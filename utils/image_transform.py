from PIL import Image
import torchvision.transforms as transforms

__all__ = ['img_transform']

new_size = (640, 640)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2, hue=0.2),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomErasing(p=0.7,scale=(0.009, 0.01)),
    transforms.ToPILImage()
])


def img_transform(img_path, mode='train',new_size=new_size,transforms=train_transform):
        image = Image.open(img_path)
        image = image.resize(new_size)
        if mode=='train':
            image = transforms(image)
        return image



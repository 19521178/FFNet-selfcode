from torchvision.models import alexnet, mobilenet_v3_large
import torchvision.transforms as transforms
import torch
from PIL import Image
import cv2


class Extractor(object):
    def forward(self, img):
        raise NotImplementedError()
    
class AlexNet(Extractor):
    def __init__(self):
        self.device = self.get_device_cuda()
        self.model = self.get_extract_model()
        self.model.to(self.device)
        
        self.transformer = self.get_transformer()
        
        self.output_size = 4096
        
    
    def get_device_cuda(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def get_extract_model(self):
        model = alexnet(pretrained=True)
        model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-2])
        model.eval()
        return model
        
    def get_transformer(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform
        
    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        # pil_img = Image.fromarray(np.uint8(img)).convert('RGB')
        img_t = self.transformer(pil_img)
        batch_t = torch.unsqueeze(img_t, 0)
        return batch_t
    
    def forward(self, frame):
        with torch.no_grad():   # thêm
            img_batch = self.preprocess(frame)
            img_batch = img_batch.to(self.device)    # thêm
            feat = self.model(img_batch)
            return feat.cpu().numpy()
        
class MobileNetV2(AlexNet):
    def __init__(self):
        super().__init__()
        self.output_size = 1280
        
    def get_extract_model(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        model.features[-1][-1] = torch.nn.Identity()
        model.classifier = torch.nn.Identity()
        model.eval()
        return model
        
class MobileNetV3Large(AlexNet):
    def __init__(self):
        super().__init__()
        self.output_size = 960
        
    def get_extract_model(self):
        model = mobilenet_v3_large(pretrained=True)
        model.classifier = torch.nn.Identity()
        model.eval()
        return model
    
    
if __name__ == '__main__':
    import numpy as np 
    import scipy.io as sio
    import cv2 as cv
    import os
    
    # videos_path = 'data/Tour20/video/'
    # save_path = 'data/Tour20/feat1/'
    # videos_path = '/content/drive/MyDrive/CS106/ffnet/train/Tour20/video/'
    # save_path = '/content/drive/MyDrive/CS106/ffnet/train/Tour20/feat_mobilev3large/'
    # videos_path = '/content/drive/MyDrive/CS106/ffnet/train/video/'
    # save_path = '/content/drive/MyDrive/CS106/ffnet/train/TVSum_MobileV3Large/feat_mobilev3large/'
    # videos_path = '/content/drive/MyDrive/CS106/ffnet/train/Summe/video/'
    # save_path = '/content/drive/MyDrive/CS106/ffnet/train/Summe/feat_alex/'
    videos_path = '/content/drive/MyDrive/CS106/ffnet/train/CoSum/video/'
    save_path = '/content/drive/MyDrive/CS106/ffnet/train/CoSum/feat_alex/'
    suffixes_feat = '_alex_fc7_feat.mat'
    
    extractor = AlexNet()
    # extractor = MobileNetV3Large()
    
    videos_filename = []
    for path, subpaths_name, files_name in os.walk(videos_path):
        videos_filename.extend(files_name)
    
    extracted_filename = os.listdir(save_path)
    extracted_filename = [filename.split('_')[0] + '.mp4' for filename in extracted_filename]
    videos_filename = [filename for filename in videos_filename if filename not in extracted_filename]
    print(videos_filename)
    # videos_name = sorted(os.listdir(videos_path))
    for filename in videos_filename:
        print(os.path.splitext(filename)[0])
        features = []
        capture = cv.VideoCapture(videos_path + filename)
        # capture = cv.VideoCapture(videos_path+filename[:2] + '/' + filename)
        success, frame = capture.read()
        while success:
            feat = extractor.forward(frame)
            features.append(feat)
            success, frame = capture.read()
        
        sio.savemat(save_path + filename.split('.')[0] + suffixes_feat, {'Features': features})
    
    
    
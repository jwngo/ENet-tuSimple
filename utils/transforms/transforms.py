import cv2

import numpy as np
import torch
from torchvision.transforms import Normalize as Normalize_th

class CustomTransform:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def __eq__(self): 
        return str(self) == name

    def __iter__(self):
        def iter_fn():
            for t in [self]:
                yield t
        return iter_fn()

    def __contains__(self, name):
        for t in self.__iter__():
            if isinstance(t, Compose):
                if name in t:
                    return True
            elif name == t:
                return True
        return False

class Compose(CustomTransform):
    """
    All transform in Compose should be able to accept two non None variable, img and boxes
    """
    def __init__(self, *transforms): 
        self.transforms = [*transforms]

    def __call__(self, sample): 
        for t in self.transforms:
            sample = t(sample)
        return sample 

    def __iter__(self):
        return iter(self.transforms)

    def modules(self):
        yield self
        for t in self.transforms:
            if isinstance(t, Compose):
                for _t in t.modules():
                    yield _t
            else: 
                yield t

class RandomCrop(CustomTransform):
    """Random crop the image & target

    Args:
        size: Expected size after cropping, (w,h)
    Notes: 
        - If the image is smaller than the crop size, returns the original image
    """
    def __init__(self, size):
        crop_size = size
        self.crop_size = crop_size 

    def __call__(self, sample): 
        img = sample.get('img')
        if img.shape[0] < self.crop_size[1] or img.shape[1] < self.crop_size[0]:
            print("Input image is smaller than expected output size!") 
            return sample 
        segLabel = sample.get('segLabel', None) 
        margin_h = max(img.shape[1] - self.crop_size[0], 0) # img is (h,w) crop_size is (w,h)
        margin_w = max(img.shape[0] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[1]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[0]

        # crop the image
        img_crop = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]

        # crop the target
        if segLabel is not None: 
            segLabel_crop = segLabel[crop_y1:crop_y2, crop_x1:crop_x2, ...]

        _sample = sample.copy()
        _sample['img'] = img_crop
        _sample['segLabel'] = segLabel_crop
        return _sample 

class Resize(CustomTransform):
    def __init__(self, size):
        self.size = size #(W,H)

    def __call__(self,sample): 
        img = sample.get('img')
        segLabel = sample.get('segLabel', None) 

        img = cv2.resize(img, self.size, interpolation=cv2.INTER_CUBIC)
        if segLabel is not None: 
            segLabel = cv2.resize(segLabel, self.size, interpolation=cv2.INTER_NEAREST)

        _sample = sample.copy() 
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample 

    def reset_size(self, size): 
        if isinstance(size, int):
            size = (size, size) 
        self.size = size

class RandomResize(Resize): 
    """ 
    Resize to (w, h), where w randomly samples from (minW, maxW) and h randomly samples from (minH, maxH) 
    """
    def __init__(self, minW, maxW, minH=None, maxH=None, batch=False): 
        if minH is None or maxH is None:
            minH, maxH = minW, maxW
        super(RandomResize, self).__init__((minW, minH))
        self.minW = minW 
        self.maxW = maxW
        self.minH = minH
        self.maxH = maxH
        self.batch = batch

    def random_set_size(self): 
        w = np.random(randint(self.minW, self.maxW+1))
        h = np.random(randint(self.minH, self.maxH+1))
        self.reset_size((w, h))

class Rotation(CustomTransform):
    def __init__(self, theta): 
        self.theta = theta

    def __call__(self, sample): 
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)  

        u = np.random.uniform() 
        degree = (u-0.5) * self.theta
        R = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), degree, 1)
        img = cv2.warpAffine(img, R, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        if segLabel is not None: 
            segLabel = cv2.warpAffine(segLabel, R, (segLabel.shape[1], segLabel.shape[0]), flags=cv2.INTER_NEAREST)

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample

    def reset_theta(self, theta):
        self.theta = theta

class Normalize(CustomTransform): 
    def __init__(self, mean, std): 
        self.transform = Normalize_th([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __call__(self, sample):
        img = sample.get('img')

        img = self.transform(img) 

        _sample = sample.copy()
        _sample['img'] = img
        return _sample

class ToTensor(CustomTransform):
    def __init__(self, dtype=torch.float):
        self.dtype=dtype

    def __call__(self, sample): 
        img = sample.get('img') 
        segLabel = sample.get('segLabel', None) 
        exist = sample.get('exist', None) 

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).type(self.dtype) / 255.
        if segLabel is not None:
            segLabel = torch.from_numpy(segLabel).type(torch.long)

        if exist is not None:
            exist = torch.from_numpy(exist).type(torch.float32) # BCEloss requires float tensor

        _sample = sample.copy() 
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        _sample['exist'] = exist
        return _sample



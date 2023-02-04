from torchvision import transform


class Transformation(object):

    def __init__(self, params):
        '''Todo: set parameters through params['..']'''
        self.transform = transform.Compose([
            transform.ToTensor(),
            transform.RandomResizedCrop(),
            transform.RandomHorizontalFlip(),
            transform.RandomRotation(),
            transform.ColorJitter(),
        ])

    def __call__(self, img):
        img = self.trans_func(img)
        return img
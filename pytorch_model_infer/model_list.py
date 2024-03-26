import timm 
from pprint import pprint
from torchvision import models 

if __name__ == '__main__':
    print("timm模型库列表:")
    model_names = timm.list_models(pretrained=True)
    pprint(model_names)
    print("torchvision模型库列表:")
    model_names = [name for name in dir(models) if not name.startswith('__') and callable(getattr(models, name))]
    print(model_names)

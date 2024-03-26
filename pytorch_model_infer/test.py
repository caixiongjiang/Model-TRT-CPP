import timm 


model = timm.create_model('resnet18.a3_in1k')
print(model.default_cfg)
model = timm.create_model('resnet34.a3_in1k')
print(model.default_cfg)
model = timm.create_model('resnet50.a3_in1k')
print(model.default_cfg)
model = timm.create_model('resnet101.a3_in1k')
print(model.default_cfg)
model = timm.create_model('resnet152.a3_in1k')
print(model.default_cfg)
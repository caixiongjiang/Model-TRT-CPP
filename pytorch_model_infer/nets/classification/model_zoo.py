class_model_zoo = {
    "resnet18": "resnet18.a3_in1k",
    "resnet34": "resnet34.a3_in1k",
    "resnet50": "resnet50.a3_in1k",
    "resnet101": "resnet101.a3_in1k",
    "resnet152": "resnet152.a3_in1k",
    "vit-t": "vit_tiny_r_s16_p8_224.augreg_in21k_ft_in1k",
    "vit-s": "vit_small_r26_s32_224.augreg_in21k_ft_in1k",
    "vit-l": "vit_large_r50_s32_224.augreg_in21k_ft_in1k",
}


class_weight_zoo = {
    "resnet18": "./data/model_zoo/classification/resnet/resnet18_a3_0-40c531c8.pth",
    "resnet34": "./data/model_zoo/classification/resnet/resnet34_a3_0-a20cabb6.pth",
    "resnet50": "./data/model_zoo/classification/resnet/resnet50_a3_0-59cae1ef.pth",
    "resnet101": "./data/model_zoo/classification/resnet/resnet101_a3_0-1db14157.pth",
    "resnet152": "./data/model_zoo/classification/resnet/resnet152_a3_0-134d4688.pth",
    "vit-t": "./data/model_zoo/classification/vit/R_Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz",
    "vit-s": "./data/model_zoo/classification/vit/R26_S_32-i21k-300ep-lr_0.001-aug_light0-wd_0.03-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.03-res_224.npz",
    "vit-l": "./data/model_zoo/classification/vit/R50_L_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz",
}


class_model_url_zoo = {
    "resnet18": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a3_0-40c531c8.pth",
    "resnet34": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a3_0-a20cabb6.pth",
    "resnet50": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a3_0-59cae1ef.pth",
    "resnet101": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a3_0-1db14157.pth",
    "resnet152": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a3_0-134d4688.pth",
    "vit-t": "https://storage.googleapis.com/vit_models/augreg/R_Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz",
    "vit-s": "https://storage.googleapis.com/vit_models/augreg/R26_S_32-i21k-300ep-lr_0.001-aug_light0-wd_0.03-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.03-res_224.npz",
    "vit-l": "https://storage.googleapis.com/vit_models/augreg/R50_L_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz",
}

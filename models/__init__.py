from .resnet import resnet18, resnet34, resnet50
from .task import Classification, Regression


def get_model(cfg, model_name=None):
    if model_name:
        feature_extractor_name = model_name
    else:
        feature_extractor_name = cfg.model.architecture
    pretrained = cfg.model.pretrained
    num_classes = cfg.dataset.num_classes
    task = cfg.model.task
    print(f"Using pretrained {pretrained}.")
    if feature_extractor_name == 'resnet18':
        feature_extractor = resnet18(pretrained)
    elif feature_extractor_name == 'resnet34':
        feature_extractor = resnet34(pretrained)
    elif feature_extractor_name == 'resnet50':
        feature_extractor = resnet50(pretrained)
    
    if task == 'cls':
        model = Classification(feature_extractor, num_classes, cfg=cfg)
    elif task == 'reg':
        parts = cfg.dataset.parts
        model = Regression(feature_extractor, num_out=len(parts)*2, cfg=cfg)
    return model

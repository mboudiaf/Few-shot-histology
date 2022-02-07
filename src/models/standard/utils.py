from torchvision import _internally_replaced_utils

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
}


def load_pretrained_weights(model, arch, load_fc=True):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    state_dict = _internally_replaced_utils.load_state_dict_from_url(model_urls[arch], progress=True)
    if load_fc:
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')
    else:
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        res = model.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == set(['fc.weight', 'fc.bias']), 'issue loading pretrained weights'
    print('Loaded pretrained weights for {}'.format(arch))
    return model

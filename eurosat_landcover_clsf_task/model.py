import torch.nn as nn
import torchvision.models as models
import torch
import config


class LULC_Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        #self.network = models.wide_resnet50_2(pretrained=True)
        
        self.network, embedding = resnet.__dict__['resnet50'](zero_init_residual=True)
        state_dict = torch.load('../vicreg_pure/vicreg/exp/resnet50_100epochs_16_factor.pth', map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
            state_dict = {key.replace("module.backbone.", ""): value for (key, value) in state_dict.items()}
        self.network.load_state_dict(state_dict, strict=False)
        
        n_inputs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
                              nn.Linear(n_inputs, 256),
                              nn.ReLU(),
                              nn.Dropout(0.5),
                              nn.Linear(256, config.NUM_CLASSES),
                              nn.LogSoftmax(dim=1)
                                )
    def forward(self, xb):
        return self.network(xb)
    
    def freeze(self):
        for param in self.network.parameters():
            param.require_grad=False
        for param in self.network.fc.parameters():
            param.require_grad=True
    def unfreeze(self):
        for param in self.network.parameters():
            param.require_grad=True
            
def get_model():
    use_cuda = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if use_cuda else 'cpu')
    model = LULC_Model()
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=lambda storage, loc: storage))

    return model
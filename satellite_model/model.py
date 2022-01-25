import torch
from torchvision.models import resnet50
import torch.nn as nn

def get_model(sources, device, checkpoint=None, dropout=None, heteroscedastic=False):
    """ Returns a model suitable for the given sources """
    if sources == "S2":
        return get_S2_no2_model(device, checkpoint, dropout, heteroscedastic)

    elif sources == "S2S5P":
        return get_S2S5P_no2_model(device, checkpoint, dropout, heteroscedastic)

def get_S2_no2_model(device, checkpoint=None, dropout=None, heteroscedastic=False):
    """ Returns a ResNet for Sentinel-2 data with a regression head """
    backbone = get_resnet_model(device, checkpoint)
    backbone.fc = nn.Identity()

    if dropout is not None:
        head = Head(2048, 512, dropout, heteroscedastic)
        head.turn_dropout_on()
    else:
        head = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 1))

    regression_model = ResnetRegressionHead(backbone, head)

    return regression_model

class Head(nn.Module):
    def __init__(self, input_dim, intermediate_dim, dropout_config, heteroscedastic):
        super(Head, self).__init__()
        self.dropout1_p = dropout_config["p_second_to_last_layer"]
        self.dropout2_p = dropout_config["p_last_layer"]
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if heteroscedastic:
            # split the output layer into [mean, sigma2]
            self.fc2 = nn.Linear(intermediate_dim, 2)
        else:
            self.fc2 = nn.Linear(intermediate_dim, 1)
        self.dropout_on = True

    def forward(self, x):
        x = nn.functional.dropout(x, p=self.dropout1_p, training=self.dropout_on)
        x = self.fc1(x)
        x = self.relu(x)
        x = nn.functional.dropout(x, p=self.dropout2_p, training=self.dropout_on)
        x = self.fc2(x)

        return x

    def turn_dropout_on(self, use=True):
        self.dropout_on = use


def get_S2S5P_no2_model(device, checkpoint=None, dropout=None, heteroscedastic=False):
    """ Returns a model with two input streams
    (one for S2, one for S5P) followed by a dense
    regression head """
    backbone_S2 = get_resnet_model(device, checkpoint)
    backbone_S2.fc = nn.Identity()

    backbone_S5P = nn.Sequential(nn.Conv2d(1, 10, 3),
                              nn.ReLU(),
                              nn.MaxPool2d(3),
                              nn.Conv2d(10, 15, 5),
                              nn.ReLU(),
                              nn.MaxPool2d(3),
                              nn.Flatten(),
                              nn.Linear(1815, 128),
                             )
    if dropout is not None:
        # add dropout to linear layers of regression head
        #head = nn.Sequential(nn.Dropout(dropout["p_second_to_last_layer"]), nn.Linear(2048+128, 544), nn.ReLU(), nn.Dropout(dropout["p_last_layer"]), nn.Linear(544, 1))
        head = Head(2048+128, 544, dropout, heteroscedastic)
        head.turn_dropout_on()
    else:
        head = nn.Sequential(nn.Linear(2048+128, 544), nn.ReLU(), nn.Linear(544, 1))

    regression_model = MultiBackboneRegressionHead(backbone_S2, backbone_S5P, head)

    return regression_model

def get_resnet_model(device, checkpoint=None):
    """
    create a resnet50 model, optionally load pretrained checkpoint
    and pass it to the device
    """
    model = resnet50(pretrained=False, num_classes=19)
    model.conv1 = torch.nn.Conv2d(12, 64, kernel_size=(3,3), stride=(2,2), padding=(3,3), bias=False)
    model.to(device)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))

    return model

class ResnetRegressionHead(nn.Module):
    """ Wrapper class to put a regression head on
    a resnet model """
    def __init__(self, backbone, head):
        super(ResnetRegressionHead, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return x

class MultiBackboneRegressionHead(nn.Module):
    """ Wrapper class that combines features extracted
    from two inputs (S2 and S5P) with a regression head """
    def __init__(self, backbone_S2, backbone_S5P, head):
        super(MultiBackboneRegressionHead, self).__init__()
        self.backbone_S2 = backbone_S2
        self.backbone_S5P = backbone_S5P
        self.head = head
        self.use_dropout = True

    def forward(self, x):
        s5p = x.get("s5p")
        x = x.get("img")

        x = self.backbone_S2(x)
        s5p = self.backbone_S5P(s5p)
        x = torch.cat((x, s5p), dim=1)
        x = self.head(x)

        return x

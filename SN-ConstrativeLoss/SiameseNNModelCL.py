import torch.nn as nn


class SiameseNetworkCL(nn.Module):
    def __init__(self, activation="relu"):
        super(SiameseNetworkCL, self).__init__()

        self.activation = activation

        self.convolutionalLayerRelu = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(10, 10)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(64, 128, kernel_size=(7, 7)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(128, 128, kernel_size=(4, 4)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(128, 256, kernel_size=(4, 4)),
            nn.ReLU(),
            nn.Flatten()
        )

        self.convolutionalLayerSelu = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(10, 10)),
            nn.SELU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(64, 128, kernel_size=(7, 7)),
            nn.SELU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(128, 128, kernel_size=(4, 4)),
            nn.SELU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(128, 256, kernel_size=(4, 4)),
            nn.SELU(),
            nn.Flatten()
        )

        self.connectedLayer = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        if self.activation == "relu":
            x = self.convolutionalLayerRelu(x)
        elif self.activation == "selu":
            x = self.convolutionalLayerSelu(x)
        x = self.connectedLayer(x)
        return x

    def forward(self, input_1, input_2):
        output_1 = self.forward_once(input_1)
        output_2 = self.forward_once(input_2)
        return output_1, output_2

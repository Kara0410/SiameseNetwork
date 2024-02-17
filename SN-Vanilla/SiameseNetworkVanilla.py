import torch
import torch.nn as nn


class SiameseNetworkVanilla(nn.Module):
    def __init__(self):
        super(SiameseNetworkVanilla, self).__init__()

        self.convolutionalLayer = nn.Sequential(
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
            nn.Flatten(),

            nn.Linear(9216, 4096),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        return self.convolutionalLayer(x)

    def forward(self, input_1, input_2):
        embedded_output_1 = self.forward_once(input_1)
        embedded_output_2 = self.forward_once(input_2)

        # L1ComponentWiseDistance = torch.abs()
        siamese_output = self.classifier(torch.abs(embedded_output_1 - embedded_output_2))
        return siamese_output

import torch
from torch import nn
from torchvision import transforms


class FurtherImprovedModel(torch.nn.Module):
    """
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS
        
        self.channels = 64
        
        """
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=30),
            transforms.RandomCrop(160),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(hue=.1, saturation=.1)
        ])
        """
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.channels*2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=self.channels*2, out_channels=self.channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.channels*2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=self.channels*2, out_channels=self.output_channels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=self.output_channels[0]),
        )

        self.conv2 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=self.output_channels[0], out_channels=self.channels*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.channels*4),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=self.channels*4, out_channels=self.output_channels[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=self.output_channels[1])
        )

        self.conv3 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=self.output_channels[1], out_channels=self.channels*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.channels*8),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=self.channels*8, out_channels=self.output_channels[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=self.output_channels[2]),
        )
        
        self.conv4 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=self.output_channels[2], out_channels=self.channels*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.channels*4),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=self.channels*4, out_channels=self.output_channels[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=self.output_channels[3])
        )

        self.conv5 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=self.output_channels[3], out_channels=self.channels*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.channels*4),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=self.channels*4, out_channels=self.output_channels[4], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=self.output_channels[4])
        )

        self.conv6 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=self.output_channels[4], out_channels=self.channels*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.channels*4),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=self.channels*4, out_channels=self.output_channels[5], kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=self.output_channels[5])
        )
        


    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        out_features.append(self.conv1(x))
        out_features.append(self.conv2(out_features[0]))
        out_features.append(self.conv3(out_features[1]))
        out_features.append(self.conv4(out_features[2]))
        out_features.append(self.conv5(out_features[3]))
        out_features.append(self.conv6(out_features[4]))
        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            out_channel = self.output_channels[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)


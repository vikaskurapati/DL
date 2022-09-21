"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl

class SegmentationNN(pl.LightningModule):
# class SegmentationNN(nn.Module):
    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO - Train Your Model                                              #
        ########################################################################

        self.cnn = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3,padding=1),
            #nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            #nn.Conv2d(32, 64, kernel_size=3,padding=1),
            #nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            #nn.ReLU(),
            #nn.Conv2d(64, 128, kernel_size=3,padding=1),
            #nn.BatchNorm2d(2*self.hparams["num_filters"], eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 192, kernel_size=3,padding=1),
            #nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=3,padding=1),
            # nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=3,padding=1),
            # nn.BatchNorm2d(4*self.hparams["num_filters"], eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.Conv2d(4*self.hparams["num_filters"], 240, kernel_size=self.hparams["kernel_size"], padding=self.hparams["padding"], stride = self.hparams["stride"]),
            # nn.BatchNorm2d(240, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(192, 384, kernel_size=3,padding=1),
            # nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3,padding=1),
            #nn.Conv2d(512, 256, kernel_size=3,padding=1),
            # nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            #nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3,padding=1),
            # # nn.BatchNorm2d(4*self.hparams["num_filters"], eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            # nn.Conv2d(240, self.hparams["num_filters"], kernel_size=self.hparams["kernel_size"], padding=self.hparams["padding"], stride = self.hparams["stride"]),
            # nn.BatchNorm2d(self.hparams["num_filters"], eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU()
        )
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        # self.upsampling = nn.Upsample(scale_factor=2, mode = 'bilinear')
        self.adjust = nn.Sequential(
            nn.Conv2d(256, 23, kernel_size=1, padding=0, stride = self.hparams["stride"]),
            nn.BatchNorm2d(23, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.ReLU()
        )

        # model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x = self.cnn(x)
        # print(x.shape)

        x = self.upsampling(x)
        # print(x.shape)

        x = self.adjust(x)

        # print(x.shape)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

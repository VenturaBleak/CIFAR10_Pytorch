import torch
import torch.nn as nn
def model_choice(model,
                 pretrained = False,
                 num_classes = None):
    """Choose model and load pretrained weights, return model, dimensions of input image
    Args:
        model (str): model name
        pretrained (bool): load pretrained weights
    Returns:
    :return model: model architecture, resolution: dimensions of input image
    """
    # raise error if num classes not specified
    if num_classes is None:
        raise ValueError("Please specify num_classes")
    # raise error if model not in list
    if model not in ['NiN']:
        raise ValueError("Model not in list")

    # specify models
    if model == 'NiN':
        class NiN(nn.Module):
            """Network in Network (NiN) model
            # https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/nin-cifar10_batchnorm.ipynb

            The CNN architecture is based on Lin, Min, Qiang Chen, and Shuicheng Yan.
            * "Network in network." arXiv preprint arXiv:1312.4400 (2013).
            This paper compares using BatchNorm before the activation function and after the activation function as it is nowadays common practice;
            as suggested in Ioffe, Sergey, and Christian Szegedy.
            * "Batch normalization: Accelerating deep network training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015)
            """
            def __init__(self, num_classes):
                super(NiN, self).__init__()
                self.num_classes = num_classes
                self.classifier = nn.Sequential(
                    nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2, bias=False),
                    nn.BatchNorm2d(192),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(160),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(96),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Dropout(0.5),

                    nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2, bias=False),
                    nn.BatchNorm2d(192),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(192),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(192),
                    nn.ReLU(inplace=True),
                    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Dropout(0.5),

                    nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(192),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(192),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.AvgPool2d(kernel_size=8, stride=1, padding=0),

                )

            def forward(self, x):
                x = self.classifier(x)
                logits = x.view(x.size(0), self.num_classes)
                probas = torch.softmax(logits, dim=1)
                return logits, probas

        model = NiN(num_classes=num_classes)
        return model, 32
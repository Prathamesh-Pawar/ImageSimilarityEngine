import torch.nn as nn

class EmbeddingNet(nn.Module):
    """
    EmbeddingNet is a simple CNN model which.
    Take a 32x32x3 image as input
    Outputs a 1x64 feature vector
    """
    def __init__(self):
        super(EmbeddingNet, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(3, 32, 5),  # 3 input channels (for RGB)
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 256),  # Adjusted input size for fully connected layer
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, 128),  # Output size of 2
            nn.PReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)



class TripletNet(nn.Module):
    """
    Triplet Net is the main Neural Network for image similarity
    It is made up of 3 EmbeddingNet networks
    """
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

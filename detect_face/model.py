import torch
import torchvision

EMBEDDING_SIZE=350

class APN_Model(torch.nn.Module):
    def __init__(self, emb_size=EMBEDDING_SIZE):
        super(APN_Model, self).__init__()

        self.model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=EMBEDDING_SIZE)

    def forward(self, images):
        embeddings = self.model(images)

        return embeddings

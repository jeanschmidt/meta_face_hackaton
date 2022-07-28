import torch
import timm  # https://timm.fast.ai/

EMBEDDING_SIZE=300

class APN_Model(torch.nn.Module):
    def __init__(self, emb_size=EMBEDDING_SIZE):
        super(APN_Model, self).__init__()

        self.efficient_net = timm.create_model("efficientnet_b0", pretrained=False)
        self.efficient_net.classifier = torch.nn.Linear(
            in_features=self.efficient_net.classifier.in_features, out_features=emb_size
        )

    def forward(self, images):
        return self.efficient_net(images)

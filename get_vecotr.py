import random
from glob import glob
import os
from os import path

import numpy as np
import pandas as pd
import timm  # https://timm.fast.ai/
import torch
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm



class APN_Model(nn.Module):
    def __init__(self, emb_size=EMBEDDING_SIZE):
        super(APN_Model, self).__init__()

        self.efficient_net = timm.create_model("efficientnet_b0", pretrained=False)
        # load pre-trained weights
        self.efficient_net.load_state_dict(torch.load("/var/svcscm/efficientnet_b0_ra-3dd342df.pth"))
        # adjust the output dimension (embeddings size)
        self.efficient_net.classifier = nn.Linear(
            in_features=self.efficient_net.classifier.in_features, out_features=emb_size
        )

    def forward(self, images):
        embeddings = self.efficient_net(images)

        return embeddings


def img_2_vec(model: APN_Model, img_path: str) -> np.ndarray:
    """Gets the path to an image of some face and returns a vector representation for this face."""
    model.eval()

    with torch.no_grad():
        img = io.imread(img_path)
        img = torch.from_numpy(img).permute(2, 0, 1) / 255.0
        vec = model(img.unsqueeze(0))

    return vec.squeeze().numpy()



if __name__ == '__main__':
    # laod the model
    some_model = model = APN_Model()
    some_model.load_state_dict(torch.load("/var/svcscm/best_model_0.1.pt"))

    # compare two images from the validation dataset
    model.eval()

    with torch.no_grad():
        for i, (ancor, pos, neg) in enumerate(df_valid.values[:20]):
            embs_ancor = img_2_vec(model, ancor)
            embs_pos = img_2_vec(model, pos)
            embs_neg = img_2_vec(model, neg)

            print(
                i,
                f"Positive: {cosine_similarity([embs_ancor], [embs_pos])[0]}",
                f"Negative: {cosine_similarity([embs_ancor], [embs_neg])[0]}",
           )

    

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


class APN_Model_Resnet(nn.Module):
    def __init__(self, emb_size=EMBEDDING_SIZE):
        super(APN_Model, self).__init__()

        # load pre-trained model
        self.model = models.resnet50(pretrained=True)

        # freeze the first 20 layers
        for i, param in enumerate(self.model.parameters()):
            if i == 20:
                break
            param.requires_grad = False

        in_features = self.model.fc.in_features # 2048
        self.model.fc = nn.Linear(in_features=in_features, out_features=EMBEDDING_SIZE)

    def forward(self, images):
        embeddings = self.model(images)

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
    some_model = model = APN_Model_Resnet()
    some_model.load_state_dict(torch.load("models/model_resnet_0.4.2.pt"))

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

    

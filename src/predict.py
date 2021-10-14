import numpy as np
from scipy.sparse import data
import torch
import joblib
from transformers import get_linear_schedule_with_warmup
import config
import dataset
import engine
from model import EntityModel

if __name__ == "__main__" :
    meta_data = joblib.load("meta.bin")
    enc_pos = meta_data["enc_pos"]
    enc_tag = meta_data["enc_tag"]

    sentence  = """
    ahmedelmogtaba is going to sudan
    """

    tokenized_sentence = config.TOKENIZER.encode(
        sentence
    )


    sentence = sentence.split()
    print(sentence)
    print(tokenized_sentence)

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    test_dataset = dataset.EntityDataSet(texts = [sentence], pos= [[0]*len(sentence)], tags=[[0]*len(sentence)])

    device = torch.device(config.DEVICE)
    model = EntityModel(num_tag = num_tag , num_pos = num_pos)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k , v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag,pos,_ = model(**data)

        print(
            enc_tag.inverse_transform(
                tag.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
        )
        print(
            enc_pos.inverse_transform(
                pos.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
        )

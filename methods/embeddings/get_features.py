from torch.utils.data import DataLoader
import torch
from tqdm import tqdm


def get_embeddings(dataset, model, tokenizer, hp):
    dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=False)
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    vectors = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Feature_Extraction'):
            sents, ids = batch
            tokens = tokenizer(list(sents), padding=True, truncation=True,return_tensors='pt')

            tokens.to(device)
            output = model(**tokens)
            vector = output[0]
            vectors += output.last_hidden_state[:,0,:].cpu().numpy().tolist()
            labels += ids.numpy().tolist()

    return vectors, labels

def get_sent_embeddings(dataset, model, hp):
    dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=False)
    vectors = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Feature_Extraction'):
            sents, ids = batch
            vector = model.encode(sents)
            vectors += vector.tolist()
            labels += ids.numpy().tolist()

    return vectors, labels

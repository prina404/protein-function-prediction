from transformers import T5Tokenizer, T5EncoderModel
from utils.config import CFG
from typing import Iterable

import torch.nn as nn
import torch
import re

class ProtTransClassifier(nn.Module):
    name = "ProtTransMLP"
    '''
    A simple MLP for classifying the ProtTrans embeddings
    '''
    def __init__(self, config: dict,):
        super().__init__()
        
        # check that the config has all the required keys
        required_keys = ['n_hidden', 'hidden_dims', 'dropout_rate', 'batch_norm']

        assert all(key in config for key in required_keys), f"Config is missing keys: {required_keys}"
        assert config['n_hidden'] == len(config['hidden_dims']), "n_hidden must be equal to the length of hidden_dims"
        
        input_dim = config['embedding_dim']
        layers = nn.ModuleList()
        
        for hidden_dim in config['hidden_dims']:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())

            if config['batch_norm']:    # batch_norm and dropout are mutually exclusive
                layers.append(nn.BatchNorm1d(hidden_dim))
            else:
                layers.append(nn.Dropout(config['dropout_rate']))
            
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, CFG['data']['num_classes'] + 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



class ProtTransEncoder():
    '''
    Encodes protein sequences using the ProtTrans T5 model.
    '''
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(CFG.device)
        self.model.eval()
        self.model.compile() 

    def encode(self, seq: Iterable[str]) -> torch.Tensor:
        '''
        Encode a sequence of protein sequences into a tensor of embeddings. Each sequence
        has one embedding associated with it. The output is a tensor of shape (batch_size, embedding_dim=1024).
        '''
        # tokenize the sequence
        seq = list(map(self._verify_protein_str, seq))
        ids = self.tokenizer(seq, return_tensors="pt", padding='longest', add_special_tokens=True)

        input_ids = (ids['input_ids']).to(CFG.device)
        attention_mask = (ids['attention_mask']).to(CFG.device)

        # compute the embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state

        B, _, E = embeddings.shape
        means = torch.zeros((B, E), dtype=torch.float32).to(CFG.device)
        for i, emb in enumerate(embeddings):
            # mean of the embeddings
            n_tokens = len(seq[i])
            means[i] = torch.mean(emb[:n_tokens], dim=0)

        prot_embeddings = means.detach().cpu()
        # free GPU mem
        del input_ids, attention_mask, outputs, embeddings, means
        torch.cuda.empty_cache()
        
        return prot_embeddings
    
    def to(self, device: str) -> None:
        self.model.to(device)
        self.device = device
    
    @staticmethod
    def _verify_protein_str(seq: str) -> str:
        '''
        Verify that the protein is in the format that the tokenizer expects:
        - Each residue is separated by a space
        - The sequence is upper case
        - The four uncommon residues, if present, are replaced by the 'X' character
        '''
        assert len(seq) > 1

        seq = seq.upper()
        seq = re.sub(r"[UZOB]", "X", seq)

        odd_chars = seq[1:][::2]
        has_spaces = all(map(lambda x: x == " ", odd_chars))
        if not has_spaces:
            seq = "".join([c + ' ' for c in seq])
        
        return seq


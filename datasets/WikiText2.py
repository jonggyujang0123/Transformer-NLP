"""
Mnist Data loader, as given in Mnist tutorial
"""
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import Tensor
from typing import Tuple
import torch
class WikiText2DataLoader:
    def __init__(self, config):
        """
        :param config:
        """
        # First, using the train data, we configure the tokenizer and vocabulary
        self.config = config
        train_iter = WikiText2('./data', split='train')
        self.tokenizer = get_tokenizer('basic_english') # Convert the sentence in a token, (e.g., "good morning!" --> ["good", "morning", "1"]
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, train_iter), specials=['<unk>']) # indexing tokenizer to train_iter, <unk> means the word shown less than 2 times
        # map : ex) filter(lambda x: x^2, range(10)) -> [0, 1, 4, 9, ..., 81]
        # reduce : reduce(lambda x,y : x+y, [0,1,2,3,4]) --> 10 (cumulative sum)
        self.vocab.set_default_index(self.vocab['<unk>'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        """Prepare the dataset"""
        train_iter, val_iter, test_iter = WikiText2()
        train_data = self.data_process(train_iter)
        val_data = self.data_process(val_iter)
        test_data = self.data_process(test_iter)
        batch_size = config.batch_size
        test_batch_size = config.test_batch_size
        self.train_data = self.batchify(train_data, batch_size)
        self.val_data = self.batchify(val_data, test_batch_size)
        self.test_data = self.batchify(test_data, test_batch_size)


    def data_process(self, raw_text_iter) -> Tensor:
        """
        # numel : total number of elements of tensor, V
        # filter(lambda t: t.numel() >0, data) --> return data element with the t.numel()>0
        Conver raw text into a flat Tensor
        """
        data=[torch.tensor(self.vocab(self.tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data))) 

    def batchify(self, data: Tensor, bsz: int) -> Tensor:
        """
        Divides the data into bsz separate sequences
        """
        seq_len = data.size(0) //bsz
        data = data[:seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        return data.to(self.device)

    def get_batch(self, source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
        bptt = 35
        """
        args: 
            source: Tensor, shape [ full_seq_len, batch_size ] 
            i:int

        Returns:
            tuple (data,target), where data has shape [seq_len, batch_size] and target has shape [seq_len, batch_size]
        """
        seq_len = min(bptt, len(source)-1-i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target
    def get_ntokens(self):
        return len(self.vocab)

    def finalize(self):
        pass


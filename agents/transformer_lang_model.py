"""
Mnist Main agent, as mentioned in the tutorial
"""
import numpy as np

from tqdm import tqdm
import shutil
import random
import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from agents.base import BaseAgent
import math
from graphs.models.transformerLanguageModel import TransformerLanguageModel
from datasets.WikiText2 import WikiText2DataLoader

from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics

cudnn.benchmark = True

import wandb
import os
import time
os.environ['WANDB_SILENT'] = "true"

class TransLangModel(BaseAgent):

    def __init__(self, config, config_dict, gpu_ids, mode):
        wandb.init(project="Pytorch-Project-Template", entity="jonggyujang0123", config=config_dict, name=f'{config.agent}-lr-{config.lr}', group=f'{config.agent}_{mode}')
        super().__init__(config)
                         
                         
        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda
        self.device = torch.device("cuda")
        # set the manual seed for torch
        self.manual_seed = self.config.seed
         # Define Dataloader
        self.data_loader = WikiText2DataLoader(config=config)
         # define models
        self.ntokens=self.data_loader.get_ntokens()
        self.model = TransformerLanguageModel(self.ntokens, config.emsize, config.nhead, config.d_hid, config.nlayers, config.dropout).to(self.device)
        self.mode = mode
        self.checkpoint_dir = config.checkpoint_dir + '/' + config.exp_name + '/' 
#         if len(gpu_ids)>1:
#             self.model = nn.DataParallel(self.model)

        # define loss
        self.loss = nn.CrossEntropyLoss()

        # define optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
#         self.load_checkpoint(self.checkpoint_dir + self.config.checkpoint_file)
        # Summary Writer

    def load_checkpoint(self, file_name="checkpoint.pth.tar"):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        example :
        """
        filename = self.checkpoint_dir + file_name
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        example
        """
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(state, self.checkpoint_dir + file_name)
        # If it is the best copy it to another file 'model_best.pth.tar'
        self.logger.info("Checkpoint saved successfully to '{}' at (epoch {})\n"
            .format(self.checkpoint_dir, self.epoch))
        if is_best:
            self.logger.info("This is the best model\n")
            shutil.copyfile(self.checkpoint_dir + file_name,
                            self.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        """
        The main operator
        :return:
        """
        if self.mode == 'test':
            self.test()
        else:
            self.train()
#            self.test()


    def train(self):
        import time
        import copy
        """
        Main training loop
        :return:
        """
        val_loss_current = float('inf')
        for self.epoch in range(1, self.config.epochs + 1):
            epoch_start_time = time.time()
            self.train_one_epoch()
            val_loss = self.validate()
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {self.epoch:3d} | time {elapsed:5.2f}s| '
                  f'valid_loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
            print('-' * 89)
            if val_loss < val_loss_current and self.config.save_best:
                self.save_checkpoint(self.config.checkpoint_file, is_best=True)
                val_loss_current = val_loss

            self.current_epoch += 1
            self.scheduler.step()
    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """
        Generate an upper-triangular matrix of -inf, with zeros on diag.
        0, -inf, -inf, 
        0,    0, -inf,
        0,    0,    0,
        """
        return torch.triu(torch.ones(sz,sz)* float('-inf'), diagonal=1)
                         
    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        bptt=35
        total_loss = 0.
        start_time = time.time()
        src_mask = self.generate_square_subsequent_mask(bptt).to(self.device)
        self.model.train()
        num_batches = self.data_loader.train_data.size(0) //bptt
        for batch, i in enumerate(range(0,self.data_loader.train_data.size(0) -1, bptt)):
            data, targets = self.data_loader.get_batch(self.data_loader.train_data, i)
            batch_size = data.size(0)
            if not batch_size == bptt: # if batchsize < bptt 
                src_mask = src_mask[:batch_size, :batch_size]       
            output = self.model(data, src_mask)
            loss = self.loss(output.view(-1,self.ntokens), targets)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            if batch % self.config.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / self.config.log_interval
                cur_loss = total_loss / self.config.log_interval
                ppl = math.exp(cur_loss)
                total_loss =0.
                self.logger.info(f'| epoch {self.epoch:3d} | {batch:5d}/{num_batches:5d} batches |'
                              f'lr {lr:04.4f} | ms/batch {ms_per_batch:2.2f} | '
                              f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            self.current_iteration += 1
        wandb.log({"loss": loss.mean().detach().item()})
        wandb.watch(self.model)
                         
    def test(self):
        """
        One cycle of model validation
        :return:
        """
        bptt=35
        self.model.eval()
        val_loss = 0
        src_mask = self.generate_square_subsequent_mask(bptt).to(device)
        with torch.no_grad():
            for i in range(0, self.data_loader.test_data.size(0) -1 , bptt):
                data, targets = self.data_loader.get_batch(self.data_loader.test_data, i)
                batch_size = data.size(0)
                if batch_size != bptt :
                    src_mask = src_mask[:batch_size, :batch_size]
                output = self.model(data, src_mask)
                output_flat = output.view(-1, self.ntokens)
                total_loss += batch_size * self.loss(output_flat, targets).item()
        print('=' * 89)
        test_ppl = math.exp(total_loss/ (self.data_loader.test_data.size(0)-1))
        print(f'| End of training | test loss {total_loss/ (self.data_loader.test_data.size(0)-1):5.2f} | '
              f'test ppl {test_ppl:8.2f}')
        print('=' * 89)
        return total_loss / (self.data_loader.test_data.size(0)-1)

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        bptt=35
        self.model.eval()
        val_loss = 0
        src_mask = self.generate_square_subsequent_mask(bptt).to(self.device)
        with torch.no_grad():
            for i in range(0, self.data_loader.val_data.size(0) -1 , bptt):
                data, targets = self.data_loader.get_batch(self.data_loader.val_data, i)
                batch_size = data.size(0)
                if batch_size != bptt :
                    src_mask = src_mask[:batch_size, :batch_size]
                output = self.model(data, src_mask)
                output_flat = output.view(-1, self.ntokens)
                total_loss += batch_size * self.loss(output_flat, targets).item()
        return total_loss / (len(val_data)-1)


    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        if not self.mode == 'test':
            self.save_checkpoint(self.config.checkpoint_file)
        self.data_loader.finalize()

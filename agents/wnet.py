"""
Main Agent for WNet
"""
import numpy as np

from tqdm import tqdm
import shutil

import torch
import torch.cuda
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from agents.base import BaseAgent
from graphs.models.wnet import WNet
from datasets.wdata import WeatherDataLoader

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList, cls_accuracy
from utils.misc import print_cuda_statistics
from utils.train_utils import adjust_learning_rate

import logging

cudnn.benchmark = True


class WNetAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.logger = logging.getLogger("Agent")
        
        # Create an instance from the model
        self.model = WNet(self.config)
        # Create an instance from the data loader
        self.data_loader = WeatherDataLoader(self.config)
        # Create an instance from the loss
        self.loss = nn.CrossEntropyLoss()
        # Create an instance from the optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr = self.config.learning_rate,
                                         momentum=float(self.config.momentum),
                                         weight_decay=self.config.weight_decay,
                                         nesterov=True)
        
        
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & self.config.cuda
        
        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed_all(self.config.seed)
            torch.cuda.set_device(self.config.gpu_device)
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
            #print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.config.seed)
            self.logger.info("Operation will be on *****CPU***** ")
            
        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)
        #self.load_checkpoint(self.config.checkpoint_file)
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='WNet')
        

    def load_checkpoint(self, filename):
        """
        Latest checkpoint loader
        :param filename: name of the checkpoint file
        :return:
        """
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            
            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint['iteration']))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, filename="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param filename: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        
        torch.save(state, self.config.checkpoint_dir + filename)
        
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            if self.config.mode == 'test':
                self.validate()
            else:
                self.train()
                
        except KeyboardInterrupt:
            self.logger.info("You have enetered CTRL+C... Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()
            
            valid_acc = self.validate()
            is_best = valid_acc > self.best_valid_acc
            if is_best:
                self.best_valid_acc = valid_acc
            self.save_checkpoint(is_best=is_best)

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        # Initialize tqdm
        tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                          desc="Epoch-{}-".format(self.current_epoch))
        # Set the model to be in training mode
        self.model.train()
        # Initialize your average meters
        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()
        top2_acc = AverageMeter()
        
        current_batch = 0
        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.cuda(), y.cuda()
                
            # current iteration over total iterations
            x, y = Variable(x), Variable(y)
            lr = adjust_learning_rate(self.optimizer, self.current_epoch, self.config, batch=current_batch,
                                      nBatch=self.data_loader.train_iterations)
            # model
            pred = self.model(x)
            # loss
            cur_loss = self.loss(pred, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError("Loss is NaN during training...")
            # optimizer
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()
            
            top1, top2 = cls_accuracy(pred.data, y.data, topk=(1, 2))
            
            epoch_loss.update(cur_loss.item())
            top1_acc.update(top1.item(), x.size(0))
            top2_acc.update(top2.item(), x.size(0))
            
            self.current_iteration += 1
            current_batch += 1
            
            self.summary_writer.add_scalar("epoch/loss", epoch_loss.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch/accuracy", top1_acc.val, self.current_iteration)
        tqdm_batch.close()
        
        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " + "loss: " +
                         str(epoch_loss.val) + "- Top1 Acc: " + str(top1_acc.val) + "- Top2 Acc: " +
                         str(top2_acc.val))

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="Validation at -{}-".format(self.current_epoch))
        
        # set the model in evaluation mode
        self.model.eval()
        
        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()
        top2_acc = AverageMeter()
        
        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.cuda(), y.cuda()
                
            x, y = Variable(x), Variable(y)
            # model
            pred = self.model(x)
            # loss
            cur_loss = self.loss(pred, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError("Loss is NaN during validation...")
                
            top1, top2 = cls_accuracy(pred.data, y.data, topk=(1, 2))
            epoch_loss.update(cur_loss.item())
            top1_acc.update(top1.item(), x.size(0))
            top2_acc.update(top2.item(), x.size(0))
        
        self.logger.info("Validation results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.avg) + "- Top1 Acc: " + str(top1_acc.val) + "- Top2 Acc: " + str(top2_acc.val))

        tqdm_batch.close()

        return top1_acc.avg

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.data_loader.finalize()

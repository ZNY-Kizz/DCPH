import torch
import time
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils import *
from modules import *
from collections import OrderedDict
from optim import create_optimizer
from scheduler import create_scheduler
import logging
import os
import importlib
from torch.utils.data import DataLoader
import math
import torch.nn.functional as F

SetSeed()

class DCPH(object):
    def __init__(self, cfg):
        self.cfg = cfg
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        print("Current CUDA device:", torch.cuda.current_device())
        print("Config device:", self.cfg.device)
        
        # Record
        self.writer = SummaryWriter(log_dir=self.cfg.tensorboard_dir, flush_secs=5)
        logging.basicConfig(level=logging.INFO,
                        filename=cfg.log_file,
                        filemode='a',
                        format='%(asctime)s: %(message)s'
                        )
        self.saver = CheckpointSaver(checkpoint_dir=cfg.save_dir)
        # Model - HAVSP Component
        self.HAA = VideoSequenceModel(self.cfg.input_dim, self.cfg.hidden_dim, self.cfg.dropout, self.cfg.frame_length)
        self.VVA = VisualAdapter(self.cfg.input_dim, self.cfg.hidden_dim, self.cfg.dropout)
        # Model - PGDH Component
        # self.PGDH = HashMapping_HyP2(self.cfg.input_dim, self.cfg.hidden_dim, self.cfg.binary_dim, self.cfg.dropout, self.cfg.num_classes)
        self.PGDH = HashMapping_CVH(self.cfg.input_dim, self.cfg.binary_dim, self.cfg.num_classes)
        self.euclidean_awl = AutomaticWeightedLoss(2)
        self.euclidean_model_list = [self.HAA, self.VVA, self.euclidean_awl]
        self.hamming_model_list = [self.PGDH]
        self.model_list = self.euclidean_model_list + self.hamming_model_list
        self.euclidean_models_params, self.euclidean_named_params = [], []
        for model in self.euclidean_model_list:
            self.euclidean_models_params.extend(model.parameters())
            self.euclidean_named_params.extend(model.named_parameters())
        self.hamming_models_params, self.hamming_named_params = [], []
        for model in self.hamming_model_list:
            self.hamming_models_params.extend(model.parameters())
            self.hamming_named_params.extend(model.named_parameters())
        # Optimizer and LR Scheduler
        self.euclidean_optimizer = create_optimizer(self.cfg, self.euclidean_models_params, self.euclidean_named_params)
        self.hamming_optimizer = create_optimizer(self.cfg, self.hamming_models_params, self.hamming_named_params)
        self.euclidean_lr_scheduler, self.euclidean_num_epochs = create_scheduler(self.cfg, self.cfg.euclidean_num_epochs ,self.euclidean_optimizer)
        self.hamming_lr_scheduler, self.hamming_num_epochs = create_scheduler(self.cfg, self.cfg.hamming_num_epochs, self.hamming_optimizer)
        if torch.cuda.is_available():
            for model in self.model_list:
                model.cuda(self.cfg.device)
            self.TextEncoder = TextEncoder(self.cfg.device)
        else:
            self.TextEncoder = TextEncoder('cpu')
    
    def euclidean_training(self):
        for model in self.euclidean_model_list:
            model.train()
    
    def euclidean_validating(self):
        for model in self.euclidean_model_list:
            model.eval()
    
    def hamming_training(self):
        for model in self.hamming_model_list:
            model.train()
    
    def hamming_validating(self):
        for model in self.hamming_model_list:
            model.eval()

    def train(self, train_loader, val_loader, test_loader):
        best_metric = None
        best_epoch = None
        for epoch in range(1, self.euclidean_num_epochs):
            logging.info("#" * 10 + f" Euclidean Train: Epoch {epoch} " + "#" * 10)
            euclidean_metrics: OrderedDict[str, int] = self.euclidean_train_epoch(epoch, train_loader)
            if (self.euclidean_lr_scheduler is not None) and (epoch < self.cfg.lr_decay_end_epoch):
                self.euclidean_lr_scheduler.step(epoch, euclidean_metrics['loss'])
        # for epoch in range(1, self.hamming_num_epochs):
            logging.info("#" * 11 + f" Hamming Train: Epoch {epoch} " + "#" * 11)
            hamming_metrics: OrderedDict[str, int] = self.hamming_train_epoch(epoch, train_loader)
            logging.info("#" * 16 + f" Val: Epoch {epoch} " + "#" * 16)
            eval_metrics: OrderedDict[str, int] = self.validate(epoch, val_loader, train_loader)
            if (self.hamming_lr_scheduler is not None) and (epoch < self.cfg.lr_decay_end_epoch):
                self.hamming_lr_scheduler.step(epoch, eval_metrics[self.cfg.eval_metrics])
            if self.saver is not None:
                save_metric = eval_metrics[self.cfg.eval_metrics]
                best_metric, best_epoch = self.saver.save_checkpoint(
                    self.model_list, self.euclidean_optimizer, self.hamming_optimizer, self.cfg, epoch=epoch, metric=save_metric)
        if best_metric is not None:
            logging.info('Train Best Metric: {0} (epoch {1})'.format(best_metric, best_epoch))
        logging.info("#" * 15 + f" Test " + "#" * 15)
        best_model_path = os.path.join(self.cfg.save_dir, "model_best.pth.tar")
        if os.path.exists(best_model_path):
            state_dict = torch.load(best_model_path, weights_only=False)['state_dict']
            self.load_state_dict(state_dict)
            test_metrics: OrderedDict[str, int] = self.test(test_loader, val_loader)
        else:
            logging.warning(f"Best model file not found: {best_model_path}, skipping test")
            logging.warning("Possible reason: no checkpoint was saved during training")
            
    def euclidean_train_epoch(self, epoch, loader):
        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.euclidean_training()
        self.hamming_validating()
        end = time.time()
        last_idx = len(loader) - 1
        for batch_idx, (frames_features, text_features, labels) in enumerate(loader):
            last_batch = batch_idx == last_idx
            data_time_m.update(time.time() - end)
            if torch.cuda.is_available():
                frames_features = torch.Tensor(frames_features).cuda(self.cfg.device)
                text_features = torch.Tensor(text_features).cuda(self.cfg.device)
                labels = torch.Tensor(labels).cuda(self.cfg.device)
            else:
                frames_features = torch.Tensor(frames_features)
                text_features = torch.Tensor(text_features)
                labels = torch.Tensor(labels)
            # HAVSP Component
            video_features = self.HAA(frames_features)
            pseudo_tokens = self.VVA(video_features)
            pseudo_text_features = self.TextEncoder.encode_with_pseudo_tokens(pseudo_tokens)
            loss = self.euclidean_loss_fn(video_features, pseudo_text_features, text_features)
            losses_m.update(loss.item(), video_features.size(0))
            self.euclidean_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.euclidean_models_params, max_norm=20, norm_type=2)
            self.euclidean_optimizer.step()
            batch_time_m.update(time.time() - end)
            if last_batch or batch_idx % self.cfg.log_interval == 0:
                lrl = [param_group['lr'] for param_group in self.euclidean_optimizer.param_groups]
                lr = sum(lrl) / len(lrl)
                logging.info('Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                      'Loss: {loss.val:.6f} ({loss.avg:.4f})  '
                      'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                      '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                      'LR: {lr:.3e}  '
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch,
                    batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=video_features.size(0) / batch_time_m.val,
                    rate_avg=video_features.size(0) / batch_time_m.avg,
                    lr=lr,
                    data_time=data_time_m))
            end = time.time()
        self.writer.add_scalar('Euclidean_AWL_Loss', losses_m.avg, epoch)
        return OrderedDict([('loss', losses_m.avg)])
    
    def hamming_train_epoch(self, epoch, loader):
        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.euclidean_validating()
        self.hamming_training()
        end = time.time()
        last_idx = len(loader) - 1
        for batch_idx, (frames_features, text_features, labels) in enumerate(loader):
            last_batch = batch_idx == last_idx
            data_time_m.update(time.time() - end)
            if torch.cuda.is_available():
                frames_features = torch.Tensor(frames_features).cuda(self.cfg.device)
                text_features = torch.Tensor(text_features).cuda(self.cfg.device)
                labels = torch.Tensor(labels).cuda(self.cfg.device)
            else:
                frames_features = torch.Tensor(frames_features)
                text_features = torch.Tensor(text_features)
                labels = torch.Tensor(labels)
            with torch.no_grad():
                # HAVSP Component
                video_features = self.HAA(frames_features)
                pseudo_tokens = self.VVA(video_features)
                pseudo_text_features = self.TextEncoder.encode_with_pseudo_tokens(pseudo_tokens)
            # PGDH Component
            pseudo_t, _ = self.PGDH(pseudo_text_features)
            loss = self.hamming_loss_fn(pseudo_t, labels)
            losses_m.update(loss.item(), video_features.size(0))
            self.hamming_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.hamming_models_params, max_norm=20, norm_type=2)
            self.hamming_optimizer.step()
            batch_time_m.update(time.time() - end)
            if last_batch or batch_idx % self.cfg.log_interval == 0:
                lrl = [param_group['lr'] for param_group in self.hamming_optimizer.param_groups]
                lr = sum(lrl) / len(lrl)
                logging.info('Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                      'Loss: {loss.val:.6f} ({loss.avg:.4f})  '
                      'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                      '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                      'LR: {lr:.3e}  '
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch,
                    batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=video_features.size(0) / batch_time_m.val,
                    rate_avg=video_features.size(0) / batch_time_m.avg,
                    lr=lr,
                    data_time=data_time_m))
            end = time.time()
        self.writer.add_scalar('Hamming_AWL_Loss', losses_m.avg, epoch)
        return OrderedDict([('loss', losses_m.avg)])

    def validate(self, epoch, query_loader, database_loader):
        self.euclidean_validating()
        self.hamming_validating()
        with torch.no_grad():
            base_b, base_label = self.predict_binary_codes(database_loader)
            query_b, query_label = self.predict_binary_codes(query_loader)
        mAP_K, _, _ = mean_average_precision(base_b, query_b, base_label, query_label, self.cfg.topK)
        logging.info('Val mAP_K  >>>>> {mAP_K:.4f}'.format(mAP_K = mAP_K))
        self.writer.add_scalar('mAP_K', mAP_K, epoch)
        return OrderedDict([('mAP_K', mAP_K)])
    
    # def test(self, query_loader, database_loader):
    #     self.euclidean_validating()
    #     self.hamming_validating()
    #     with torch.no_grad():
    #         base_b, base_label = self.predict_binary_codes(database_loader)
    #         query_b, query_label = self.predict_binary_codes(query_loader)
    #     mAP_K, _, _ = mean_average_precision(base_b, query_b, base_label, query_label, self.cfg.topK)
    #     logging.info('Test mAP_K  >>>>> {mAP_K:.4f}'.format(mAP_K = mAP_K))
    #     return OrderedDict([('mAP_K', mAP_K)])
    def test(self, query_loader, database_loader):
        self.hamming_validating()
        with torch.no_grad():
            base_b, base_label = self.predict_binary_codes(database_loader)
            query_b, query_label = self.predict_binary_codes(query_loader)
        
        Dist = HammingDist(query_b, base_b)
        Rel = CalRel(query_label, base_label)
        
        mAP_K, _, _ = mean_average_precision(base_b, query_b, base_label, query_label, self.cfg.topK)
        acg_K = average_cumulative_gain(Dist, Rel, self.cfg.topK)
        ndcg_K = normalized_discounted_cumulative_gain(Dist, Rel, self.cfg.topK)
        wap_K = weighted_average_precision(Dist, Rel, self.cfg.topK)
        test_result_msg = 'Test mAP_K  >>>>> {mAP_K:.4f}'.format(mAP_K = mAP_K)
        logging.info(test_result_msg)
        print(f"{test_result_msg}", flush=True)
        
        acg_msg = 'Test ACG_K  >>>>> {acg_K:.4f}'.format(acg_K = acg_K)
        logging.info(acg_msg)
        print(f"{acg_msg}", flush=True)
        
        ndcg_msg = 'Test NDCG_K  >>>>> {ndcg_K:.4f}'.format(ndcg_K = ndcg_K)
        logging.info(ndcg_msg)
        print(f"{ndcg_msg}", flush=True)
        
        wap_msg = 'Test WAP_K  >>>>> {wap_K:.4f}'.format(wap_K = wap_K)
        logging.info(wap_msg)
        print(f"{wap_msg}", flush=True)
        
        return OrderedDict([('mAP_K', mAP_K), ('ACG_K', acg_K), ('NDCG_K', ndcg_K), ('WAP_K', wap_K)])
    
    def test_all(self, query_loader, database_loader):
        self.euclidean_validating()
        self.hamming_validating()
        with torch.no_grad():
            base_b, base_label = self.predict_binary_codes(database_loader)
            query_b, query_label = self.predict_binary_codes(query_loader)
        Dist = HammingDist(query_b, base_b)
        Rel = CalRel(query_label, base_label)
        mAP_K, _, _ = mean_average_precision(base_b, query_b, base_label, query_label, self.cfg.topK)
        print('Test mAP_K  >>>>> {mAP_K:.4f}'.format(mAP_K = mAP_K))
        acg_K = average_cumulative_gain(Dist, Rel, self.cfg.topK)
        print('Test ACG_K  >>>>> {acg_K:.4f}'.format(acg_K = acg_K))
        ndcg_K = normalized_discounted_cumulative_gain(Dist, Rel, self.cfg.topK)
        print('Test NDCG_K  >>>>> {ndcg_K:.4f}'.format(ndcg_K = ndcg_K))
        wap_K = weighted_average_precision(Dist, Rel, self.cfg.topK)    
        print('Test WAP_K  >>>>> {wap_K:.4f}'.format(wap_K = wap_K))


    def predict_binary_codes(self, loader):
        self.euclidean_validating()
        self.hamming_validating()
        all_codes = []
        all_labels = []
        with torch.no_grad():
            for idx, (frames_features, text_features, labels) in enumerate(loader):
                if torch.cuda.is_available():
                    frames_features = torch.Tensor(frames_features).cuda(self.cfg.device)
                else:
                    frames_features = torch.Tensor(frames_features)
                video_features = self.HAA(frames_features)
                pseudo_tokens = self.VVA(video_features)
                pseudo_text_features = self.TextEncoder.encode_with_pseudo_tokens(pseudo_tokens)
                _, b = self.PGDH(pseudo_text_features)
                all_codes.append(b)
                all_labels.append(labels)
        return torch.cat(all_codes, 0).cpu().detach().numpy(), torch.cat(all_labels, 0).cpu().detach().numpy()

    def euclidean_loss_fn(self, video_features, pseudo_text_features, text_features):
        # Dual-Constraint Contrastive Alignment: L_bias + L_confusion
        cos_criterion = nn.CosineEmbeddingLoss()
        if torch.cuda.is_available():
            cos_criterion_target = torch.as_tensor([1], device=self.cfg.device)
        else:
            cos_criterion_target = torch.as_tensor([1], device="cpu")
        L_bias = cos_criterion(video_features, text_features, cos_criterion_target)
        L_confusion = cos_criterion(text_features, pseudo_text_features, cos_criterion_target)
        loss = self.euclidean_awl(L_bias, L_confusion)
        return loss
    
    def cal_distance(self, ti, tj):
        inner_product = ti @ tj.t()
        norm = ti.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ tj.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        cos = inner_product / norm.clamp(min=0.0001)
        return (1 - cos.clamp(max=0.99)) * self.cfg.binary_dim / 2
    
    def hamming_loss_fn(self, t, labels):
        loss = self.PGDH.cal_loss(t, labels)
        return loss
    
    def state_dict(self):
        return [model.module.state_dict() if hasattr(model, 'module') else model.state_dict() for model in self.model_list]
    
    def load_state_dict(self, state_dict):
        for i in range(len(self.model_list)):
            new_state_dict = OrderedDict()
            for k, v in state_dict[i].items():
                new_state_dict[k] = v
            self.model_list[i].load_state_dict(new_state_dict, strict=True)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = "config"
    my_config = importlib.import_module(config_path)
    Config = getattr(my_config, 'Config')
    cfg = Config()
    cfg.tensorboard_dir = os.path.join(current_dir, "tensorboard")
    cfg.log_file = os.path.join(current_dir, "test.log")
    cfg.save_dir = os.path.join(current_dir, "test")
    model = DCPH(cfg)
    train_set = MyDataset(os.path.join(current_dir, "datasets/train.json"), os.path.join(current_dir, "data"))
    val_set = MyDataset(os.path.join(current_dir, "datasets/val.json"), os.path.join(current_dir, "data"))
    test_set = MyDataset(os.path.join(current_dir, "datasets/test.json"), os.path.join(current_dir, "data"))
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers, pin_memory=True)
    model.train(train_loader, val_loader, test_loader)
    state_dict = torch.load(os.path.join(current_dir, "test/model_best.pth.tar"))['state_dict']
    model.load_state_dict(state_dict)
    model.validate(1, test_loader, val_loader)


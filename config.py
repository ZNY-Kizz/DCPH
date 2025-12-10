class Config():
    def __init__(self):
        # Save Configs
        self.log_interval = 25 # batch

        # DataLoader
        self.batch_size = 128
        self.shuffle= True
        self.num_workers = 8
        
        # Model Configs
        self.input_dim = 512
        self.hidden_dim = 4 * 512
        self.binary_dim = 64 # [16, 32, 64, 128, 256]
        self.dropout = 0.5
        self.frame_length = 16
        self.num_classes = 82
        # self.num_classes = 63

        # Train Configs
        self.device = "cuda:0"
        self.lr = 0.0001
        self.euclidean_num_epochs = 100
        self.hamming_num_epochs = 100
        self.lr_decay_end_epoch = 50
        self.eval_metrics = "mAP_K"
        
        # Optimizer
        self.optimizer = "adam"
        self.weight_decay = 0.001
        self.momentum = 0.9
        self.opt_eps = 1e-8

        # Scheduler
        self.scheduler = "cosine"
        self.decay_epochs = [4, 8, 12, 17, 30]
        self.decay_rate = 0.94
        self.warmup_lr = 0.0001
        self.warmup_epochs = 3

        # Val Configs
        self.topK = 100
import os
from datetime import datetime
import importlib
import shutil
import argparse
from model import *

def main(binary_dim=None, train_json=None, val_json=None, test_json=None, npz_dir=None):
    current_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_path)
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M")
    my_config = importlib.import_module("config")
    Config = getattr(my_config, 'Config')
    cfg = Config()
    if binary_dim is not None:
        cfg.binary_dim = binary_dim
    assert cfg.binary_dim in [16, 32, 64, 128, 256], "Binary Dimension should be 16, 32, 64, 128 or 256!"
    work_dir = os.path.join(current_folder, 'work_dirs', str(cfg.binary_dim), formatted_time)
    tensorboard_dir = os.path.join(work_dir, "tensorboard1")
    log_dir = os.path.join(work_dir, "log1")
    save_dir = os.path.join(work_dir, "checkpoints1")
    config_dir = os.path.join(work_dir, "config1")
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    shutil.copy(os.path.join(current_folder, "config.py"), os.path.join(config_dir, f"{formatted_time}.py"))
    cfg.tensorboard_dir = tensorboard_dir
    cfg.log_file = f"{log_dir}/{formatted_time}.log"
    cfg.save_dir = save_dir
    model = DCPH(cfg)
    
    # Dataset paths - use command line arguments if provided, otherwise use defaults
    if train_json is None:
        train_json = os.path.join(current_folder, "datasets/train.json")
    if val_json is None:
        val_json = os.path.join(current_folder, "datasets/val.json")
    if test_json is None:
        test_json = os.path.join(current_folder, "datasets/test.json")
    if npz_dir is None:
        npz_dir = os.path.join(current_folder, "data")
    
    train_set = MyDataset(train_json, npz_dir)
    cfg.num_train = len(train_set)
    val_set = MyDataset(val_json, npz_dir)
    test_set = MyDataset(test_json, npz_dir)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers, pin_memory=True)
    model.train(train_loader, val_loader, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DCPH Model (Deep Contrastive Hashing with Proxy Guidance)')
    parser.add_argument('--binary_dim', type=int, default=None, 
                        choices=[16, 32, 64, 128, 256],
                        help='Binary dimension (16, 32, 64, 128, or 256)')
    parser.add_argument('--train_json', type=str, default=None,
                        help='Path to training JSON file')
    parser.add_argument('--val_json', type=str, default=None,
                        help='Path to validation JSON file')
    parser.add_argument('--test_json', type=str, default=None,
                        help='Path to test JSON file')
    parser.add_argument('--npz_dir', type=str, default=None,
                        help='Path to NPZ features directory')
    args = parser.parse_args()
    main(binary_dim=args.binary_dim, 
         train_json=args.train_json,
         val_json=args.val_json,
         test_json=args.test_json,
         npz_dir=args.npz_dir)
    
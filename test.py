import os
from datetime import datetime
import importlib
import shutil
import argparse
from model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test DCPH Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint directory (e.g., work_dirs/64/TIMESTAMP)')
    parser.add_argument('--val_json', type=str, default=None,
                        help='Path to validation JSON file')
    parser.add_argument('--test_json', type=str, default=None,
                        help='Path to test JSON file')
    parser.add_argument('--npz_dir', type=str, default=None,
                        help='Path to NPZ features directory')
    args = parser.parse_args()
    
    config_list = [args.model_path]
    for c in config_list:
        print("#"*10, c.split('/')[1], "#"*10)
        config_path = c + "/config1/" + c.split('/')[-1]
        config_path = config_path.replace("/",".")
        my_config = importlib.import_module(config_path)
        Config = getattr(my_config, 'Config')
        cfg = Config()
        cfg.tensorboard_dir = "./tensorboard"
        cfg.log_file = "./test.log"
        cfg.save_dir = "./test"
        cfg.device = "cpu"
        model = DCPH(cfg)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        if args.val_json is None:
            val_json = os.path.join(current_dir, "datasets/val.json")
        else:
            val_json = args.val_json
        if args.test_json is None:
            test_json = os.path.join(current_dir, "datasets/test.json")
        else:
            test_json = args.test_json
        if args.npz_dir is None:
            npz_dir = os.path.join(current_dir, "data")
        else:
            npz_dir = args.npz_dir
            
        val_set = MyDataset(val_json, npz_dir)
        test_set = MyDataset(test_json, npz_dir)
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers, pin_memory=True)
        state_dict = torch.load(c+"/checkpoints1/model_best.pth.tar", map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict)
        model.test_all(test_loader, val_loader)


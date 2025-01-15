

class Args(object):
    def __init__(self):
        self.data_root = "./LRS3_preprocess"
        self.video_data_root = "/jumbo/jinlab/datasets/lrs3"
        self.checkpoint_dir = "./ckpts"
        self.checkpoint_path = None
        self.use_cuda = True
        self.multi_gpu = False
        self.device_ids = [0, 1, 2, 3]
        self.epochs = 2
        self.num_workers = 8
        self.prefetch_factor = 8
        self.persistent_workers = True
        self.pin_memory = True
        self.checkpoint_interval = 5000
 


class Args(object):
    def __init__(self):
        self.data_root = "../LRS2_preprocess/main"
        self.video_data_root = "/gpudata1/datasets/XD/LRS2/mvlrs_v1/main"
        self.checkpoint_dir = "./ckpts/ckpt2/ckpt3"
        self.checkpoint_path = "./ckpts/ckpt2/mode_1_checkpoint_step000015000.pth"
        self.use_cuda = True
        self.multi_gpu = True
        self.device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        self.epochs = 100
        self.num_workers = 16
        self.prefetch_factor = 16
        self.persistent_workers = True
        self.pin_memory = True
        self.checkpoint_interval = 5000
        self.batch_size = 256


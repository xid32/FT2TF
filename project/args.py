

class Args(object):
    def __init__(self):
        self.dataset = "lrs2"
        # self.dataset = "lrs3"
        self.data_root = "../LRS2_preprocess/main"
        # self.data_root = "../LRS3_preprocess"
        self.video_data_root = "/jumbo/jinlab/XD/LRS2/mvlrs_v1/main"
        # self.video_data_root = "/jumbo/jinlab/datasets/lrs3"
        self.checkpoint_dir = "/jumbo/jinlab/XD/face_expression/project/ckpts/ckpt_lrs2_vit_small_test"
        # self.checkpoint_path = "/jumbo/jinlab/XD/face_expression/project/ckpts/ckpt0929/mode_ED_checkpoint_step000036360.pth"
        self.checkpoint_path = None
        # self.disc_checkpoint_path = "/jumbo/jinlab/XD/face_expression/project/ckpts/ckpt0929/mode_disc_checkpoint_step000036360.pth"
        self.disc_checkpoint_path = None
        self.text_source = "GT"
        self.use_cuda = True
        self.multi_gpu = False
        self.device_ids = [0, 1]
        self.test_only = True
        self.epochs = 10000
        self.num_workers = 2
        self.prefetch_factor = 2
        self.persistent_workers = True
        self.pin_memory = True
        self.checkpoint_interval = 101
        self.batch_size = 16
        self.input_frames = 15
        self.output_frames = 15


        # train step
        self.syncnet_wt = 0.03
        self.disc_model_p_loss_wt = 0.07

        # ablation exp
        self.global_exp = False
        self.local_exp = False
        self.vit_small = False
        self.vit_large = False


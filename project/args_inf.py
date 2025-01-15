

class Args(object):
    def __init__(self):
        self.sample_path = "./inference_data/sample"
        self.inf_len = 15
        self.save_path = "./inference_data"
        self.use_cuda = True
        self.use_exit_roberta = False
        self.use_exit_gpt = True
        self.checkpoint_path = "./ckpts/mode_ED_checkpoint_step000038784.pth"


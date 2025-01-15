from train_args import Args
import torch

args = Args()
def _load(checkpoint_path):
    # if args.use_cuda:
    #     checkpoint = torch.load(checkpoint_path)
    # else:
    #     checkpoint = torch.load(checkpoint_path,
    #                             map_location=lambda storage, loc: storage)
    checkpoint = torch.load(checkpoint_path,
                                 map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            s = checkpoint["optimizer"]
            new_s = {}
            for k, v in s.items():
                new_s[k.replace('module.', '')] = v
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model
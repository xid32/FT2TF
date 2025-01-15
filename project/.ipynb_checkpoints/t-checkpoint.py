from args import Args
from dataset_utils.dataset_final import Dataset
from torch.utils import data as data_utils


args = Args()
print("Init Dataset...")
train_dataset = Dataset(args, 'train')
print("Init DataLoader...")
train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers, pin_memory=args.pin_memory)
print(len(train_dataset), len(train_data_loader))
for step, (x, y, mel, text_embedding_f, encoded_texts) in enumerate(train_data_loader):
    print(x.shape, y.shape, mel.shape, text_embedding_f.shape, encoded_texts.shape)
    print("------------")
    break


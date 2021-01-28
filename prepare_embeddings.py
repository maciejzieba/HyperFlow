from datasets import get_datasets, synsetid_to_cate
from args import get_args
from pprint import pprint
from metrics.evaluation_metrics import EMD_CD
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics
from collections import defaultdict
from models.networks import HyperPointFlow
import os
import torch
import numpy as np
import torch.nn as nn


def get_loader(args, split):
    if split=='train':
        tr_dataset, _ = get_datasets(args)
    else:
        _, tr_dataset = get_datasets(args)
    if args.resume_dataset_mean is not None and args.resume_dataset_std is not None:
        mean = np.load(args.resume_dataset_mean)
        std = np.load(args.resume_dataset_std)
        tr_dataset.renormalize(mean, std)
    loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)
    return loader


def get_embeddings(model, args, split='train'):
    loader = get_loader(args, split)
    embeddings = []
    for data in loader:
        idx_b, te_pc, cat = data['idx'], data[split + '_points'], data['cate_idx']
        te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)
        z = model.encode(te_pc).cpu().detach().numpy()
        embeddings.append(z)
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


def main(args):
    model = HyperPointFlow(args)
    model = model.cuda()
    split = 'test'
    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()
    embeddings = get_embeddings(model, args, split)
    save_dir = os.path.dirname(args.resume_checkpoint)
    np.save(os.path.join(save_dir, "Shapenet_embeddings_" + split + ".npy"), embeddings)


if __name__ == '__main__':
    args = get_args()
    main(args)

from datasets import get_datasets, synsetid_to_cate
from args import get_args
from pprint import pprint
from metrics.evaluation_metrics import EMD_CD
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics
from collections import defaultdict
from models.networks import HyperPointFlow
from models.atlas_networks import HyperNetwork, TargetNetwork
import os
import torch
import numpy as np
import torch.nn as nn

from train_atlas import tile

def get_test_loader(args):
    _, te_dataset = get_datasets(args)
    if args.resume_dataset_mean is not None and args.resume_dataset_std is not None:
        mean = np.load(args.resume_dataset_mean)
        std = np.load(args.resume_dataset_std)
        te_dataset.renormalize(mean, std)
    loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)
    return loader


def evaluate_recon(model, args):
    if 'all' in args.cates:
        cates = list(synsetid_to_cate.values())
    else:
        cates = args.cates
    all_results = {}
    cate_to_len = {}
    save_dir = os.path.dirname(args.resume_checkpoint)
    for cate in cates:
        args.cates = [cate]
        loader = get_test_loader(args)

        all_sample = []
        all_ref = []
        for data in loader:
            idx_b, tr_pc, te_pc = data['idx'], data['train_points'], data['test_points']
            te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)
            tr_pc = tr_pc.cuda() if args.gpu is None else tr_pc.cuda(args.gpu)
            B, N = te_pc.size(0), te_pc.size(1)
            out_pc = model.reconstruct(tr_pc, num_points=N)
            m, s = data['mean'].float(), data['std'].float()
            m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
            s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
            out_pc = out_pc * s + m
            te_pc = te_pc * s + m

            all_sample.append(out_pc)
            all_ref.append(te_pc)

        sample_pcs = torch.cat(all_sample, dim=0)
        ref_pcs = torch.cat(all_ref, dim=0)
        cate_to_len[cate] = int(sample_pcs.size(0))
        print("Cate=%s Total Sample size:%s Ref size: %s"
              % (cate, sample_pcs.size(), ref_pcs.size()))

        # Save it
        np.save(os.path.join(save_dir, "%s_out_smp.npy" % cate),
                sample_pcs.cpu().detach().numpy())
        np.save(os.path.join(save_dir, "%s_out_ref.npy" % cate),
                ref_pcs.cpu().detach().numpy())

        results = EMD_CD(sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True)
        results = {
            k: (v.cpu().detach().item() if not isinstance(v, float) else v)
            for k, v in results.items()}
        pprint(results)
        all_results[cate] = results

    # Save final results
    print("="*80)
    print("All category results:")
    print("="*80)
    pprint(all_results)
    save_path = os.path.join(save_dir, "percate_results.npy")
    np.save(save_path, all_results)

    # Compute weighted performance
    ttl_r, ttl_cnt = defaultdict(lambda: 0.), defaultdict(lambda: 0.)
    for catename, l in cate_to_len.items():
        for k, v in all_results[catename].items():
            ttl_r[k] += v * float(l)
            ttl_cnt[k] += float(l)
    ttl_res = {k: (float(ttl_r[k]) / float(ttl_cnt[k])) for k in ttl_r.keys()}
    print("="*80)
    print("Averaged results:")
    pprint(ttl_res)
    print("="*80)

    save_path = os.path.join(save_dir, "results.npy")
    np.save(save_path, all_results)

def get_squares(N):
    sq = torch.Tensor([[0.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 0.0],
                       [1.0, 1.0]]).float().cuda()
    sq = torch.cat(N*[sq])
    return sq


def evaluate_gen(model, atlas, args):
    loader = get_test_loader(args)
    all_sample = []
    all_ref = []
    for idx, data in enumerate(loader):
        #if idx < 1:
        idx_b, te_pc = data['idx'], data['test_points']
        te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)
        B, N = te_pc.size(0), te_pc.size(1)
        z, out = model.sample(B, int(N/4), gpu=args.gpu)
        target_networks_weights = atlas(z)
        out_pc = torch.zeros((out.shape[0], 4*out.shape[1], out.shape[2])).cuda()
        for j, target_network_weight in enumerate(target_networks_weights):
            pc = out[j].detach()
            atlas_input = get_squares(int(N/4))
            x_temp = torch.cat([tile(pc, dim=0, n_tile=4), atlas_input], dim=1)
            target_network = TargetNetwork(args.zdim, target_network_weight).cuda()
            out_pc[j] = target_network(x_temp)
        # denormalize
        m, s = data['mean'].float(), data['std'].float()
        m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
        s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        all_sample.append(out_pc)
        all_ref.append(te_pc)

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    print("Generation sample size:%s reference size: %s"
          % (sample_pcs.size(), ref_pcs.size()))

    # Save the generative output
    save_dir = os.path.dirname(args.resume_checkpoint)
    np.save(os.path.join(save_dir, "model_out_smp.npy"), sample_pcs.cpu().detach().numpy())
    np.save(os.path.join(save_dir, "model_out_ref.npy"), ref_pcs.cpu().detach().numpy())
    #

    # Compute metrics
    results = compute_all_metrics(sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True)
    results = {k: (v.cpu().detach().item()
                   if not isinstance(v, float) else v) for k, v in results.items()}
    pprint(results)

    sample_pcl_npy = sample_pcs.cpu().detach().numpy()
    ref_pcl_npy = ref_pcs.cpu().detach().numpy()
    jsd = JSD(sample_pcl_npy, ref_pcl_npy)
    print("JSD:%s" % jsd)


def main(args):
    model = HyperPointFlow(args)
    model = model.cuda()
    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()
    atlas = HyperNetwork(args)
    atlas = atlas.cuda()
    print("Resume Path:%s" % args.resume_atlas)
    checkpoint = torch.load(args.resume_atlas)
    atlas.load_state_dict(checkpoint['model'])
    # vars = [0.01, 0.001, 0.0001, 0.0]
    vars = [0.01]
    for var in vars:
        print("Evaluating metrics for variance equal: %s" % var)
        model.eval()
        model.var = var
        with torch.no_grad():
            evaluate_gen(model, atlas, args)


if __name__ == '__main__':
    args = get_args()
    main(args)

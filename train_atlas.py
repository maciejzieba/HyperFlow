from datasets import ShapeNetEmbeddings, init_np_seed
from args import get_args
from models.atlas_networks import HyperNetwork, TargetNetwork
from models.networks import HyperPointFlow
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
import torch.nn as nn
import imageio
from utils import visualize_point_clouds, save
from sklearn.neighbors import KNeighborsClassifier
#from external.earth_mover_distance import EMD
from external.chamfer_loss import ChamferLoss
import time
from scipy.spatial import Delaunay

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)


def generate_points_from_uniform_distribution(size, low=-1, high=1, norm=True):
    if norm:
        while True:
            points = torch.zeros([size[0] * 3, *size[1:]]).uniform_(low, high)
            points = points[torch.norm(points, dim=1) < 1]
            if points.shape[0] >= size[0]:
                return points[:size[0]]
    else:
        return torch.zeros([size[0], *size[1:]]).uniform_(low, high)


def main(args):
    # command line args
    image_path = os.path.join(os.path.dirname(args.resume_checkpoint),'images')
    os.makedirs(image_path, exist_ok=True)
    model_path = os.path.join(os.path.dirname(args.resume_checkpoint), 'model')
    os.makedirs(model_path, exist_ok=True)
    model = HyperPointFlow(args)
    model = model.cuda()
    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()
    model.var = args.stop_var
    embeddings = ShapeNetEmbeddings(os.path.dirname(args.resume_checkpoint))
    # train_loader = DataLoader(dataset=embeddings, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.num_workers, drop_last=True, worker_init_fn=init_np_seed)
    train_loader = DataLoader(dataset=embeddings, batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    num_points = 2048
    atlas_2d_points = 25

    atlas = HyperNetwork(args)
    atlas = atlas.cuda()
    opt = atlas.make_optimizer(args)

    reconstruction_loss = ChamferLoss().cuda()
    for epoch in range(args.epochs):
        print("Epoch starts:")
        for bidx, emb in enumerate(train_loader):
            #if bidx < 10:
            opt.zero_grad()
            start_time = time.time()
            inputs = emb.cuda(args.gpu, non_blocking=True)
            _, out = model.decode(inputs, num_points)
            target_networks_weights = atlas(inputs)
            x_rec = torch.zeros((out.shape[0], out.shape[1]*atlas_2d_points, out.shape[2])).cuda()
            gt_points = []
            gen_points = []
            regularization_loss = []
            atlas_input = generate_points_from_uniform_distribution(size=(out.shape[1] * atlas_2d_points, 2),
                                                                    low=0, high=1, norm=False).cuda()
            faces = torch.zeros(out.shape[1], 2 * atlas_2d_points - 2, out.shape[2])
            for z, points in enumerate(atlas_input.reshape(out.shape[1], atlas_2d_points, 2)):
                simplices = torch.from_numpy(Delaunay(points.cpu().numpy()).simplices)
                faces[z, :simplices.shape[0]] = simplices
            for j, target_network_weight in enumerate(target_networks_weights):
                clf = KNeighborsClassifier(atlas_2d_points + 1)
                pc = out[j].detach()
                clf.fit(pc.cpu().numpy(), np.ones(len(pc)))
                x_temp = torch.cat([tile(pc, dim=0, n_tile=atlas_2d_points),atlas_input], dim=1)
                target_network = TargetNetwork(args.zdim, target_network_weight).cuda()
                x_rec[j] = target_network(x_temp)
                nearest_points = clf.kneighbors(pc.cpu().numpy(), return_distance=False)
                x_rec_nearest_points = pc[nearest_points[:, 1:].reshape(-1)].reshape((pc.shape[0], atlas_2d_points, pc.shape[1]))
                x_temp2 = x_rec[j].reshape((pc.shape[0], atlas_2d_points, pc.shape[1]))
                # Regularization
                edges = torch.cat([faces[:, :, :2], faces[:, :, 1:], faces[:, :, (0, 2)]], 1).long()
                regularization_loss.append( torch.sum(
                        torch.norm(x_temp2[-1, edges[:, :, 0], :] - x_temp2[-1, edges[:, :, 1], :], dim=2),
                        dim=1))
                gt_points.append(x_rec_nearest_points)
                gen_points.append(x_temp2)
            gt_points = torch.cat(gt_points, dim=0)
            gen_points = torch.cat(gen_points, dim=0)
            regularization_loss = torch.cat(regularization_loss, dim=0)
            loss_rec = reconstruction_loss(gt_points, gen_points)/gt_points.shape[0]
            loss_reg = regularization_loss.mean()
            loss = loss_rec + 0.1*loss_reg
            loss.backward(retain_graph=True)
            opt.step()
            duration = time.time() - start_time
            print("[Rank %d] Epoch %d Batch [%2d/%2d] Time [%3.2fs] Reconstruction loss %2.5f "
                  "Regularization loss %2.5f Total loss %2.5f "
                  % (args.rank, epoch, bidx, len(train_loader), duration,
                     loss_rec.item(), loss_reg.item(), loss.item()))

        results = []
        x_rec = x_rec[:, torch.randperm(x_rec.size()[1])]
        for idx in range(min(10, inputs.size(0))):
            res = visualize_point_clouds(out[idx], x_rec[idx,:num_points], idx,
                                         pert_order=[0, 2, 1])
            results.append(res)
        res = np.concatenate(results, axis=1)
        imageio.imsave(os.path.join(image_path, 'tr_vis_conditioned_epoch%d-gpu%s.png' % (epoch, args.gpu)), res.transpose((1, 2, 0)))

        if (epoch + 1) % args.save_freq == 0:
            save(atlas, opt, epoch + 1,
                 os.path.join(model_path, 'checkpoint-%d.pt' % epoch))
            save(atlas, opt, epoch + 1,
                 os.path.join(model_path, 'checkpoint-latest.pt'))


if __name__ == '__main__':
    args = get_args()
    main(args)
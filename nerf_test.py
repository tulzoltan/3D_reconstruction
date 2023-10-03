import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


from nerf_model import NerfModel, render_rays, set_path


@torch.no_grad()
def test(hn, hf, ray_oris, ray_dirs, device='cpu', chunk_size=10, n_bins=192, H=400, W=400):
    """
    Parameters:
        hn: distance from near plane
        hf: distance from far plane
        ray_oris: ray origins for each pixel in the image
        ray_dirs: ray directions for each pixel in the image
        dataset: ray origins and directions for generating new views
        device: device to be used for testing (gpu or cpu)
        chunk_size: separate image into chunks for memory efficiency
        n_bins: number of bins for density estimation
        H: image height
        W: image width
    """
    
    data = []
    for i in range(int(np.ceil(H/chunk_size))):
        #iterate over chunks
        ray_oris_ = ray_oris[i*W*chunk_size: (i+1)*W*chunk_size].to(device)
        ray_dirs_ = ray_dirs[i*W*chunk_size: (i+1)*W*chunk_size].to(device)
        regenerated_px_vals = render_rays(model, ray_oris_, ray_dirs_, 
                                          hn=hn, hf=hf, n_bins=n_bins)
        data.append(regenerated_px_vals)

    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    return img


if __name__ == "__main__":
    import json

    HEIGHT = {}
    WIDTH = {}
    datafiles = {}
    num_images = {}
    with open("metadata.json") as mf:
        meta = json.load(mf)
        camera_names = list(meta.keys())
        for cname in camera_names:
            HEIGHT[cname] = meta[cname]["image_height"]
            WIDTH[cname] = meta[cname]["image_width"]
            num_images[cname] = meta[cname]["number_of_images"]
            datafiles[cname] = meta[cname]["file_names"]

    camera_names = ["F_MIDLONGRANGECAM_CL"]

    #output image directory
    output_dir = set_path("novel_views")

    #model weight directory
    wgt_dir = set_path("weights")

    #parameters
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    HIDDEN_DIM = 256 #256 #1st
    NEAR = 1
    FAR = 10
    BATCH_SIZE = 1024
    NUM_BINS = 96 #192 #2nd

    load_name = f"monocam_big_BASE_HD{HIDDEN_DIM}_NB{NUM_BINS}_N{NEAR}_F{FAR}"


    #load data
    print("Loading dataset ...")
    dataset = np.empty((0, 9), dtype=np.float32)
    for cname in camera_names:
        for train_name in datafiles[cname]:
            dataset = np.vstack((dataset,
                                 np.load(train_name,
                                         allow_pickle=True)))
        print(f"{cname} data loaded")
    ds_test = torch.from_numpy(dataset)

    #set up NN model
    print("Loading neural network ...")
    model = NerfModel(hidden_dim=HIDDEN_DIM).to(DEVICE)
    model.eval()

    #load weights
    load_file = os.path.join(wgt_dir, load_name+".pth.tar")
    if os.path.exists(load_file):
        model.load_state_dict(torch.load(load_file)["state_dict"])
    else:
        print(f"File {load_file} not found.")
        import sys
        sys.exit()

    trans = np.array([0.5, 0., 0.], dtype=np.float32)
    rot = np.eye(3)

    #test model
    print("Testing ...")
    for cname in camera_names:
        tH = HEIGHT[cname]
        tW = WIDTH[cname]
        N = num_images[cname]
        img_index = np.random.randint(0, N)

        ray_oris = ds_test[img_index*tH*tW: (img_index+1)*tH*tW,  :3]
        ray_dirs = ds_test[img_index*tH*tW: (img_index+1)*tH*tW, 3:6]
        img0     = ds_test[img_index*tH*tW: (img_index+1)*tH*tW, 6: ]
        img0 = img0.reshape((tH, tW, 3))

        #generate image
        img1 = test(hn=NEAR, hf=FAR, ray_oris=ray_oris, ray_dirs=ray_dirs, 
                    device=DEVICE, n_bins=NUM_BINS, H=tH, W=tW)

        #perturb origins and directions
        ray_oris_p = ray_oris + trans
        ray_dirs_p = ray_dirs

        img2 = test(hn=NEAR, hf=FAR, ray_oris=ray_oris_p, ray_dirs=ray_dirs_p, 
                    device=DEVICE, n_bins=NUM_BINS, H=tH, W=tW)

        f, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(img0)
        ax[0].set_title("original image")
        ax[1].imshow(img1)
        ax[1].set_title("reproduced image")
        ax[2].imshow(img2)
        ax[2].set_title("perturbed image")
        plot_name = os.path.join(output_dir, f"test_{cname}_IMG{img_index}_N{NEAR}_F{FAR}.png")
        plt.savefig(plot_name, bbox_inches="tight")
        plt.close()


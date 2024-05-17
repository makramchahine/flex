import os
import sys
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from einops import rearrange

# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import modello
from model_loader import modello, device
from tqdm import tqdm
from analysis.utils import *


def blip_extract(**kwargs):
    inp = kwargs["input"]
    features = modello.net.extractor.custom_extract_multi_features_ver0(inp["image"], inp["text"])
    features = features.multimodal_embeds[:, 0]  # torch.Size([257, 768])
    features = features[1:]  # torch.Size([256, 768])
    out = features.view(1, 16, 16, features.shape[-1])  # torch.Size([1, 16, 16, 768])
    out = out.permute(0, 3, 1, 2)  # torch.Size([1, 768, 16, 16])

    return out, features


def lin_extract(**kwargs):
    inp = kwargs["input"]
    out = modello.net.extractor.last_linear_layer(inp)  # torch.Size([1, 64, 16, 16])
    features = out[0].view(64, 16 * 16).T
    return out, features


def lin_emb(**kwargs):
    inp = kwargs["input"]
    x = modello.net.policy.model.to_patch_embedding.forward(inp)  # torch.Size([1, 16, 16, 128])
    pe = posemb_sincos_2d(x)
    if kwargs["n_step"] == 3:
        # do the swap on the positional embeddings
        pe = pe.reshape(16, 16, -1)
        pe = pe.cpu().detach().numpy()
        pe = crop_and_swap(pe, SR, ER, SC, EC, mode="features", idx=kwargs["idx"], n_swaps=kwargs["n_swaps"])
        pe = torch.tensor(pe).to(device)
        # reshape to 256,128
        pe = pe.reshape(16 * 16, -1)

    out = rearrange(x, 'b ... d -> b (...) d') + pe  # torch.Size([1, 256, 128])
    return out, out[0]


def att_1(**kwargs):
    x = kwargs["input"]
    attn, ff = modello.net.policy.model.transformer.layers[0]
    x = attn(x) + x  # torch.Size([1, 256, 128])
    return x, x[0]


def lin_1(**kwargs):
    x = kwargs["input"]
    attn, ff = modello.net.policy.model.transformer.layers[0]
    x = ff(x) + x  # torch.Size([1, 256, 128])
    return x, x[0]


def att_2(**kwargs):
    x = kwargs["input"]
    attn, ff = modello.net.policy.model.transformer.layers[1]
    x = attn(x) + x  # torch.Size([1, 256, 128])
    return x, x[0]


def lin_2(**kwargs):
    x = kwargs["input"]
    attn, ff = modello.net.policy.model.transformer.layers[1]
    x = ff(x) + x  # torch.Size([1, 256, 128])
    return x, x[0]


def att_3(**kwargs):
    x = kwargs["input"]
    attn, ff = modello.net.policy.model.transformer.layers[2]
    x = attn(x) + x  # torch.Size([1, 256, 128])
    return x, x[0]


def lin_3(**kwargs):
    x = kwargs["input"]
    attn, ff = modello.net.policy.model.transformer.layers[2]
    x = ff(x) + x  # torch.Size([1, 256, 128])
    return x, x[0]


OPERATIONS = [blip_extract, lin_extract, lin_emb, att_1, lin_1, att_2, lin_2, att_3, lin_3]


def safe_flip(features, n_step, idx, n_swaps=4):
    if n_step == 0:  # OG image
        img = np.array(features)
        features = crop_and_swap(img, SR, ER, SC, EC, mode="image", idx=idx, n_swaps=n_swaps)
        out = features
        out = Image.fromarray(out)
        out = out.convert('RGB').resize((224, 224))

    elif n_step == 1:  # out of blip2
        # reshape
        features = features.reshape(16, 16, -1)
        # detach to numpy array on cpu
        features = np.array(features.cpu())
        features = crop_and_swap(features, SR, ER, SC, EC, mode="features", idx=idx, n_swaps=n_swaps)
        # features to tensor
        features = torch.tensor(features).to(device)
        out = features.view(1, 16, 16, features.shape[-1])  # torch.Size([1, 16, 16, 768])
        out = out.permute(0, 3, 1, 2)  # torch.Size([1, 768, 16, 16])


    elif n_step == 2:  # out of lin_extract
        features = features.reshape(16, 16, -1)
        features = features.cpu().detach().numpy()
        features = crop_and_swap(features, SR, ER, SC, EC, mode="features", idx=idx, n_swaps=n_swaps)
        features = torch.tensor(features).to(device)
        out = features.unsqueeze(0)
        out = out.permute(0, 3, 1, 2)

    else:  # all the way from lin_emb to lin_3
        # reshape to 16x16,dim
        features = features.reshape(16, 16, -1)
        # convert to numpy array
        features = features.cpu().detach().numpy()
        features = crop_and_swap(features, SR, ER, SC, EC, mode="features", idx=idx, n_swaps=n_swaps)
        # reshape to 256,dim
        features = features.reshape(16 * 16, -1)
        features = torch.tensor(features).to(device)
        # expand batch dimension
        out = features.unsqueeze(0)

    return out, features


def flip_step(img, txt, n_step, idx=0, n_swp=4):
    preops = OPERATIONS[:n_step]
    postops = OPERATIONS[n_step:]

    feat_list = []
    features = None

    if n_step == 0:
        out, features = safe_flip(img, n_step, idx)
        img = out
        # img should be converted from out numpy array to PIL image
        out = transforms.ToTensor()(out).to(device)
        out = {"image": out, "text": txt}

    else:
        out = transforms.ToTensor()(img).to(device)
        out = {"image": out, "text": txt}
        for op in preops:
            out, features = op(input=out, n_step=n_step, idx=idx, n_swaps=n_swp)
            feat_list.append(features)

        # flip the current features
        out, features = safe_flip(features, n_step, idx, n_swaps=n_swp)
        # replace last feature with flipped feature
        feat_list[-1] = features
        # convert numpy array to tensor
        out = out.clone().detach()

    for op in postops:
        out, features = op(input=out, n_step=n_step, idx=idx, n_swaps=n_swp)
        feat_list.append(features)

    # GETTING THE OUTPUT
    x = out.mean(dim=1)  # torch.Size([1, 128])
    x = modello.net.policy.model.to_latent(x)  # torch.Size([1, 128])
    out = modello.net.policy.model.linear_head(x)  # torch.Size([1, 4])
    # detach into a 1x4 numpy array
    out = out.cpu().detach().numpy()
    out = out[0]

    return img, feat_list, out


if __name__ == "__main__":
    texts = ["reach the watermelon"]
    inst = texts[0]
    # list all images in the directory
    images = ["/home/makramchahine/repos/fm_flight/analysis/img_test/00001760.png"]

    if not os.path.exists("results"):
        os.makedirs("results")

    LAYERS = ["OG IMG", "BLIP2", "LIN_EXT", "LIN_EMB", "ATT_1", "LIN_1", "ATT2", "LIN 2", "ATT 3", "LIN 3"]
    CLUSTEZ = "manual"

    SC = 7
    EC = 11
    SR = 7
    ER = 11

    image = Image.open(images[0])
    # resize the image to 224x224 and convert to RGB
    image = image.resize((224, 224)).convert('RGB')

    layer = 1
    n_swaps = 4
    for layer in tqdm(range(0, 4)):
        OUTS = []
        for k in range(n_swaps ** 2):
            imag, feats, output = flip_step(image, texts[0], layer, idx=k, n_swp=n_swaps)
            OUTS.append(output)

            fig, axs = plt.subplots(3, 10, figsize=(30, 20))
            axs[0, 0].imshow(imag)
            for i in range(16):
                axs[0, 0].axhline(y=i * imag.size[1] / 16, color='y', linestyle='-', linewidth=0.5)
                axs[0, 0].axvline(x=i * imag.size[0] / 16, color='y', linestyle='-', linewidth=0.5)
            axs[0, 0].axis('off')
            axs[0, 0].set_title("Image")

            for i, feat in enumerate(feats):
                feat = feat.reshape(16 * 16, -1)
                # normalize the features
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
                # send to cpu and detach
                if hasattr(feat, "cpu"):
                    feat = feat.cpu().detach().numpy()

                # get best kmeans
                KM = best_kmeans(feat, min_k=2, max_k=5, method=CLUSTEZ)
                KM.fit(feat)
                cluster_labels = KM.labels_.reshape(16, 16)
                cluster_labels = np.roll(cluster_labels, -1, axis=1)
                axs[0, i + 1].imshow(cluster_labels, cmap='Dark2')
                axs[0, i + 1].axis('off')
                # get number of clusters
                clustez = len(np.unique(KM.labels_))
                axs[0, i + 1].set_title(f"{LAYERS[i + 1]}: {clustez} Clusters")

                pca_results, tsne_results = analyze_features(feat, seed=42)
                KM_TSNE = best_kmeans(tsne_results, min_k=2, max_k=5, method=CLUSTEZ)
                KM_TSNE.fit(tsne_results)
                cluster_labels = np.array(KM_TSNE.labels_)
                axs[1, i + 1] = fig.add_subplot(3, 10, 12 + i, projection='3d')
                axs[1, i + 1].scatter3D(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=cluster_labels,
                                        cmap='Dark2')
                axs[1, i + 1].set_title("t-SNE")
                cluster_labels = KM_TSNE.labels_.reshape(16, 16)
                cluster_labels = np.roll(cluster_labels, -1, axis=1)
                axs[2, i + 1].imshow(cluster_labels, cmap='Dark2')
                axs[2, i + 1].axis('off')
                clustez = len(np.unique(KM_TSNE.labels_))
                axs[2, i + 1].set_title(f"{clustez} Clusters")

            for ax in plt.gcf().get_axes():
                ax.set_aspect('equal', 'box')
                plt.tight_layout()

        #     text = inst.replace(" ", "-")
        #     file_name = f"{text}_{LAYERS[layer]}_{CLUSTEZ}.png"
        #     # make a folder named like the filename
        #     if not os.path.exists(f"results/{text}_{LAYERS[layer]}_{CLUSTEZ}"):
        #         os.makedirs(f"results/{text}_{LAYERS[layer]}_{CLUSTEZ}")
        #     file_name = file_name.replace(".png", f"_{k}.png")
        #     # add folder to the file name
        #     file_name = f"results/{text}_{LAYERS[layer]}_{CLUSTEZ}/{file_name}"
        #     plt.gcf()
        #     # save the figure in the folder
        #     plt.savefig(file_name, dpi=300)
        #     # kill the figure
        #     plt.close()
        #     print(f"Saved {file_name}")
        #
            plt.show()

        # # make a new figure that is a 2D heatmap of the output[-1]
        # OUTS = np.array(OUTS)
        # # save the output to a file
        # np.save(f"results/{text}_{LAYERS[layer]}_{CLUSTEZ}/outputs.npy", OUTS)
        # # get the yaw rates
        # yaw = OUTS[:, -1]
        # # if not 1D, reshape to 1D
        # if len(yaw.shape) > 1:
        #     yaw = yaw.reshape(-1)
        # # reshape to square by getting sqrt
        # n = int(np.sqrt(yaw.shape[0]))
        # yaw = yaw.reshape(n, n)
        # # plot the heatmap
        # plt.imshow(yaw, cmap='viridis')
        # plt.colorbar()
        # # set max min to [-0.15, 0.15]
        # plt.clim(-0.15, 0.15)
        # plt.title("Yaw Rates")
        # plt.savefig(f"results/{text}_{LAYERS[layer]}_{CLUSTEZ}/yaw_rates.png", dpi=300)
        # # plt.show()
        # plt.close()
        #
        # # do the same for the vz values
        # vzs = OUTS[:, 2]
        # # if not 1D, reshape to 1D
        # if len(vzs.shape) > 1:
        #     vzs = vzs.reshape(-1)
        # # reshape to square by getting sqrt
        # n = int(np.sqrt(vzs.shape[0]))
        # vzs = vzs.reshape(n, n)
        # # plot the heatmap
        # plt.imshow(vzs, cmap='viridis')
        # plt.colorbar()
        # # set max min to [-0.15, 0.15]
        # plt.clim(-0.08, 0.08)
        # plt.title("Vz velocities")
        # plt.savefig(f"results/{text}_{LAYERS[layer]}_{CLUSTEZ}/vz_velocities.png", dpi=300)
        # # plt.show()
        # plt.close()

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from einops import rearrange
from torchvision import transforms


def crop_and_swap(image, start_row, end_row, start_col, end_col, mode="image", idx=None, n_swaps=4):
    # Crop the image at the specified region
    # the rows and columns correspond to regions of 1/16 of the image size so multiply by 16
    if mode == "image":
        multiplier = 14
    else:
        multiplier = 1

    width = end_col - start_col
    height = end_row - start_row

    start_row *= multiplier
    end_row *= multiplier
    start_col *= multiplier
    end_col *= multiplier

    patch = image[start_row:end_row, start_col:end_col]

    images = []

    # swap with regions starting from the top left corner and moving right and down to the bottom right corner
    for i in range(n_swaps):
        swap_start_row = min(int(i/(n_swaps-1) * (16 - height)), 16 - height)
        swap_end_row = (swap_start_row + height) * multiplier
        swap_start_row *= multiplier
        for j in range(n_swaps):
            swap_start_col = min(int(j/(n_swaps-1) * (16 - width)), 16 - width) 
            swap_end_col = (swap_start_col + width) * multiplier
            swap_start_col *= multiplier
            # Swap the cut-out patch with the specified region
            image_copy = image.copy()
            image_copy[start_row:end_row, start_col:end_col] = image[swap_start_row:swap_end_row, swap_start_col:swap_end_col]
            image_copy[swap_start_row:swap_end_row, swap_start_col:swap_end_col] = patch
            images.append(image_copy)

    if idx is not None:
        return images[idx]

    return images


def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)


def analyze_features(data, seed=42):
    tsne = TSNE(n_components=3, random_state=seed)
    tsne_results = tsne.fit_transform(data)

    pca = PCA(n_components=3, random_state=seed)
    pca_results = pca.fit_transform(data)

    return tsne_results, pca_results


def best_kmeans(X, min_k=2, max_k=8, method="elbow"):
    if method == "silhouette":
        best_k = min_k
        best_score = -1
        best_kmeans = None

        for n_clusters in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_k = n_clusters
                best_kmeans = kmeans

        print(f"Best number of clusters: {best_k} with a Silhouette Score of {best_score}")
        return best_kmeans

    elif method == "elbow":
        distortions = []
        k_values = range(min_k, max_k + 1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
            kmeans.fit(X)
            distortions.append(kmeans.inertia_)  # Inertia: Sum of squared distances to closest cluster center

        # Find the optimal k using the knee point of the elbow curve
        knee = np.gradient(np.gradient(distortions)).argmax()
        optimal_k = k_values[knee]
        print(f"Optimal number of clusters: {optimal_k}")

        return KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    elif method == "manual":
        return KMeans(n_clusters=max_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    else:
        raise ValueError("Method not recognized")
    

def step_by_step(img, text, modello, debug=False):
    img = img.convert('RGB').resize((224, 224))
    img = transforms.ToTensor()(img)

    input = {"image": img, "text": text}

    # EXTRACTOR NETWORK
    features = modello.net.extractor.custom_extract_multi_features_ver0(img, text)
    features = features.multimodal_embeds[:, 0] # torch.Size([257, 768])
    out = features[1:].view(1, 16, 16, features.shape[-1]) # torch.Size([1, 16, 16, 768])
    out = out.permute(0, 3, 1, 2) # torch.Size([1, 768, 16, 16])
    out = modello.net.extractor.last_linear_layer(out) # torch.Size([1, 64, 16, 16])

    data = features[1:, :]
    data2 = out[0].view(64, 16*16).T

    feats = [data, data2]

    # POLICY NETWORK
    x = modello.net.policy.model.to_patch_embedding.forward(out) # torch.Size([1, 16, 16, 128])
    pe = posemb_sincos_2d(x)
    x = rearrange(x, 'b ... d -> b (...) d') + pe # torch.Size([1, 256, 128])
    feats.append(x[0])
    for attn, ff in modello.net.policy.model.transformer.layers:
        x = attn(x) + x # torch.Size([1, 256, 128])
        feats.append(x[0])
        x = ff(x) + x # torch.Size([1, 256, 128])
        feats.append(x[0])

    # GETTING THE OUTPUT
    x = x.mean(dim=1) # torch.Size([1, 128])
    x = modello.net.policy.model.to_latent(x) # torch.Size([1, 128])
    out = modello.net.policy.model.linear_head(x) # torch.Size([1, 4])
    # detach into a 1x4 numpy array
    out = out.cpu().detach().numpy()
    print(out)

    if debug:
        # get the output from the entire network
        our = modello.forward(input)
        # dict to numpy array
        our = np.array([our[key].cpu().detach().numpy() for key in our.keys()])
        # transpose to match the output
        our = our.T
        # assert that the shapes are the same
        assert our.shape == out.shape, f"Shapes are not the same: {our.shape} != {out.shape}"
        # assert that all the outputs are the same if not print the difference
        assert np.allclose(our, out) == True, f"Outputs are not the same: {our} != {out}"

    return feats
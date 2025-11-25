import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
import torch

from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from transformers import CLIPProcessor, CLIPModel

# 1. Setup and Loading Model
def load_clip_model():
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# 2. Data Preparation (Helper function to get images)
def get_sample_images():
    print("Downloading sample images...")
    # List of image URLs - varied categories
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",  # Dice
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",  # Cat
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_March_2010-1.jpg/1200px-Cat_March_2010-1.jpg",  # Another Cat
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Siam_lilacpoint.jpg/1200px-Siam_lilacpoint.jpg",  # Siamese Cat
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/June_odd-eyed-cat.jpg/1200px-June_odd-eyed-cat.jpg",  # Odd-eyed Cat
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Polar_Bear_-_Alaska_%28cropped%29.jpg/1200px-Polar_Bear_-_Alaska_%28cropped%29.jpg",  # Polar Bear
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Hot_dog_with_mustard.png/1200px-Hot_dog_with_mustard.png",  # Hot dog
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Eq_it-na_pizza-margherita_sep2005_sml.jpg/1200px-Eq_it-na_pizza-margherita_sep2005_sml.jpg"  # Pizza
    ]

    images = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    for url in urls:
        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Failed to download {url}: {e}")

    return images

# 3. Get Image Embeddings
def get_image_embeddings(model, processor, images):
    print("Generating embeddings...")
    inputs = processor(images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    # Normalize embeddings
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.numpy()

# 4. Decompose (PCA)
def decompose_embeddings_pca(embeddings, n_components=2):
    print(f"Decomposing embeddings using PCA to {n_components} components...")
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings, pca

# 5. Clustering (K-Means)
def cluster_embeddings(embeddings, n_clusters=3):
    print(f"Clustering embeddings into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans

# 6. Text Embeddings for Cluster Naming
def get_text_embeddings(model, processor, texts):
    print("Generating text embeddings for labels...")
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.numpy()

def name_clusters(model, processor, kmeans, candidate_labels):
    print("Naming clusters using CLIP text encoder...")
    # Get embeddings for candidate labels
    text_features = get_text_embeddings(model, processor, candidate_labels)

    # Get cluster centers from KMeans (these are in the image embedding space)
    # Shape: (n_clusters, 512)
    cluster_centers = kmeans.cluster_centers_

    # Normalize cluster centers to make them comparable to normalized text embeddings
    centers_tensor = torch.tensor(cluster_centers)
    centers_tensor = centers_tensor / centers_tensor.norm(p=2, dim=-1, keepdim=True)

    # Compute cosine similarity: (n_clusters, 512) @ (512, n_labels) -> (n_clusters, n_labels)
    similarity = (centers_tensor @ torch.tensor(text_features).T).numpy()

    cluster_names = {}
    for i in range(len(cluster_centers)):
        best_label_idx = np.argmax(similarity[i])
        best_label = candidate_labels[best_label_idx]
        cluster_names[i] = best_label
        print(f"Cluster {i} identified as: {best_label}")

    return cluster_names

# 7. Visualization with Images
def visualize_results(reduced_embeddings, labels, images, cluster_names, title="CLIP Embeddings Analysis"):
    print("Visualizing results with image thumbnails...")
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    unique_labels = np.unique(labels)
    palette = sns.color_palette("viridis", len(unique_labels))

    # Draw invisible points first to set up the legend
    for i, label_id in enumerate(unique_labels):
        cluster_name = cluster_names[label_id]
        plt.scatter([], [], color=palette[i], label=f"{cluster_name}", s=100)

    # Plot images at their coordinates
    for i, (x, y) in enumerate(reduced_embeddings):
        # Create thumbnail
        img = images[i].copy()
        img.thumbnail((100, 100))  # Resize for plot

        # Determine zoom level based on coordinate range to keep images readable but not huge
        # (This is a heuristic)
        imagebox = OffsetImage(img, zoom=0.3)

        # Color the border based on cluster
        ab = AnnotationBbox(imagebox, (x, y),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.3,
                            bboxprops=dict(edgecolor=palette[labels[i]], linewidth=2))
        ax.add_artist(ab)

    # Adjust plot limits to make sure images fit
    x_min, x_max = reduced_embeddings[:, 0].min(), reduced_embeddings[:, 0].max()
    y_min, y_max = reduced_embeddings[:, 1].min(), reduced_embeddings[:, 1].max()
    margin_x = (x_max - x_min) * 0.2
    margin_y = (y_max - y_min) * 0.2
    plt.xlim(x_min - margin_x, x_max + margin_x)
    plt.ylim(y_min - margin_y, y_max + margin_y)

    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster Type", loc="upper right")
    plt.grid(True, alpha=0.3)

    output_file = "clip_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    plt.close()

def main():
    # Execute the pipeline
    model, processor = load_clip_model()
    images = get_sample_images()

    if not images:
        print("No images loaded. Exiting.")
        return

    # 1. Get Embeddings
    embeddings = get_image_embeddings(model, processor, images)

    # 2. PCA
    reduced_embeddings_2d, pca = decompose_embeddings_pca(embeddings, n_components=2)

    # 3. Clustering
    # We have Cats, Polar Bear, Food, Dice.
    # Let's try 3 clusters (maybe Animals, Food, Object)
    labels, kmeans = cluster_embeddings(embeddings, n_clusters=3)

    # 4. Name Clusters
    candidate_labels = ["animal", "food", "object", "vehicle", "person"]
    cluster_names = name_clusters(model, processor, kmeans, candidate_labels)

    # 5. Visualize
    visualize_results(reduced_embeddings_2d, labels, images, cluster_names)

    print("\nExplained Variance Ratio of PCA components:", pca.explained_variance_ratio_)

if __name__ == "__main__":
    main()

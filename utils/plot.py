import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

from sklearn.metrics import silhouette_samples


def line_graph(X, y, _path):
    plt.figure()
    plt.plot(X, y)
    plt.savefig(_path)


def silhouette_plot(X, y, k, silhouette_score, centers, _path):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (k + 1) * 10])
    sample_silhouette_values = silhouette_samples(X, y)

    y_lower = 10
    for i in range(k):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[y == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_score, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(y.astype(float) / k)
    ax2.scatter(
        X[:, 0],
        X[:, 1],
        marker=".",
        s=30,
        lw=0,
        alpha=0.7,
        c=colors,
        edgecolor="k",
    )

    # Labeling the clusters
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker="o", c="white", alpha=1, s=200, edgecolor="k")

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        ("Silhouette analysis for KMeans clustering on sample data " f"with n_clusters = {k}"),
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(_path)
    plt.close()


def clustering_plot(_path, X_range, inertia_list, silhouette_list):
    plt.figure()
    plt.plot(X_range, inertia_list, color="red")
    plt.ylabel("Inertia")
    plt.savefig(os.path.join(_path, "inertia.png"))
    plt.clf()
    plt.plot(X_range, silhouette_list, color="blue")
    line = 10
    for i in silhouette_list[::-1]:
        if i < 0.5:
            line -= 1
        else:
            break
    line = 2 if line == 1 else line
    plt.axvline(x=line, color="red", linestyle="--")
    plt.ylabel("Silhouette")
    plt.savefig(os.path.join(_path, "silhouette.png"))
    plt.close()
    _, ax1 = plt.subplots()
    ax1.plot(X_range, inertia_list, color="red")
    ax1.set_ylabel("Inertia")
    ax2 = ax1.twinx()
    ax2.plot(X_range, silhouette_list, color="blue")
    ax2.set_ylabel("Silhouette")
    plt.savefig(os.path.join(_path, "score.png"))
    plt.close()


def clustering_plot_all(_path, X_range, inertia_matrix, silhouette_matrix, scaler_name):
    color_list = ["Blue", "Orange", "Green", "Purple", "Yellow", "Pink"]
    plt.figure()
    for i, (inertia, scaler) in enumerate(zip(inertia_matrix, scaler_name)):
        plt.plot(X_range, inertia, color=color_list[i], label=scaler)

    plt.ylabel("Inertia")
    plt.legend()
    plt.savefig(os.path.join(_path, "merge_inertia.png"))
    plt.clf()

    for i, (silhouette, scaler) in enumerate(zip(silhouette_matrix, scaler_name)):
        plt.plot(X_range, silhouette, color=color_list[i], label=scaler)
        
    silhouette_max = max([max(v) for v in silhouette_matrix])
    silhouette_min = min([min(v) for v in silhouette_matrix])
    
    for threshold in [0.3, 0.5, 0.7] : 
        if silhouette_max > threshold and silhouette_min < threshold : 
            plt.axhline(y=threshold, color="red", linestyle="--")

    plt.ylabel("Silhouette")
    plt.legend()
    plt.savefig(os.path.join(_path, "merge_silhouette.png"))
    plt.close()

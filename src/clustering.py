"""
MicroSegment - Segmentation Modelling Module
Author: Senior Data Science & Rural Finance Analytics Consultant
Purpose: Compare K-Means, GMM, Agglomerative clustering; select optimal k via
         multiple validation metrics; test stability via bootstrap ARI;
         visualize with PCA and UMAP.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score
)
from sklearn.decomposition import PCA

# Try importing UMAP; fall back gracefully
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[INFO] umap-learn not installed. UMAP plots will be skipped. Install with: pip install umap-learn")


# ─────────────────────────────────────────────────────────────────────
# 1. Metric Sweep: Evaluate k = 2..10 for K-Means, GMM, Agglomerative
# ─────────────────────────────────────────────────────────────────────

def evaluate_clustering_metrics(X: np.ndarray, k_range: range = range(2, 11),
                                random_state: int = 42) -> pd.DataFrame:
    """
    For each k in k_range, fit K-Means, GMM, and Agglomerative (Ward),
    then compute Silhouette, Davies-Bouldin, Calinski-Harabasz, Inertia,
    and BIC/AIC (GMM only).
    """
    records = []
    for k in k_range:
        print(f"  Evaluating k={k} ...")

        # --- K-Means ---
        km = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=random_state)
        km_labels = km.fit_predict(X)
        records.append({
            "k": k, "algorithm": "KMeans",
            "silhouette": silhouette_score(X, km_labels),
            "davies_bouldin": davies_bouldin_score(X, km_labels),
            "calinski_harabasz": calinski_harabasz_score(X, km_labels),
            "inertia": km.inertia_,
            "bic": np.nan, "aic": np.nan
        })

        # --- GMM (full covariance) ---
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=5, max_iter=500, random_state=random_state)
        gmm_labels = gmm.fit_predict(X)
        records.append({
            "k": k, "algorithm": "GMM",
            "silhouette": silhouette_score(X, gmm_labels),
            "davies_bouldin": davies_bouldin_score(X, gmm_labels),
            "calinski_harabasz": calinski_harabasz_score(X, gmm_labels),
            "inertia": np.nan,
            "bic": gmm.bic(X), "aic": gmm.aic(X)
        })

        # --- Agglomerative (Ward linkage) ---
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        agg_labels = agg.fit_predict(X)
        records.append({
            "k": k, "algorithm": "Agglomerative",
            "silhouette": silhouette_score(X, agg_labels),
            "davies_bouldin": davies_bouldin_score(X, agg_labels),
            "calinski_harabasz": calinski_harabasz_score(X, agg_labels),
            "inertia": np.nan,
            "bic": np.nan, "aic": np.nan
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────
# 2. Metric Visualisation (Elbow, Silhouette, DB, CH, BIC/AIC)
# ─────────────────────────────────────────────────────────────────────

def plot_metric_sweep(metrics_df: pd.DataFrame, fig_dir: str = "reports/figures"):
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Cluster Validation Metrics Across Algorithms (k = 2–10)",
                 fontsize=17, weight="bold", y=0.98)

    palette = {"KMeans": "#2b5c8f", "GMM": "#8f2b6a", "Agglomerative": "#2b8f5c"}

    # Silhouette (higher = better)
    for algo in ["KMeans", "GMM", "Agglomerative"]:
        sub = metrics_df[metrics_df["algorithm"] == algo]
        axes[0, 0].plot(sub["k"], sub["silhouette"], "o-", label=algo, color=palette[algo], lw=2)
    axes[0, 0].set_title("Silhouette Score (↑)", fontsize=13, weight="semibold")
    axes[0, 0].set_xlabel("k"); axes[0, 0].set_ylabel("Score"); axes[0, 0].legend()
    axes[0, 0].axvline(x=5, ls="--", alpha=0.4, color="red", label="k=5")

    # Davies-Bouldin (lower = better)
    for algo in ["KMeans", "GMM", "Agglomerative"]:
        sub = metrics_df[metrics_df["algorithm"] == algo]
        axes[0, 1].plot(sub["k"], sub["davies_bouldin"], "o-", label=algo, color=palette[algo], lw=2)
    axes[0, 1].set_title("Davies-Bouldin Index (↓)", fontsize=13, weight="semibold")
    axes[0, 1].set_xlabel("k"); axes[0, 1].set_ylabel("Score"); axes[0, 1].legend()
    axes[0, 1].axvline(x=5, ls="--", alpha=0.4, color="red")

    # Calinski-Harabasz (higher = better)
    for algo in ["KMeans", "GMM", "Agglomerative"]:
        sub = metrics_df[metrics_df["algorithm"] == algo]
        axes[0, 2].plot(sub["k"], sub["calinski_harabasz"], "o-", label=algo, color=palette[algo], lw=2)
    axes[0, 2].set_title("Calinski-Harabasz Index (↑)", fontsize=13, weight="semibold")
    axes[0, 2].set_xlabel("k"); axes[0, 2].set_ylabel("Score"); axes[0, 2].legend()
    axes[0, 2].axvline(x=5, ls="--", alpha=0.4, color="red")

    # Elbow (K-Means inertia)
    km_sub = metrics_df[metrics_df["algorithm"] == "KMeans"]
    axes[1, 0].plot(km_sub["k"], km_sub["inertia"], "o-", color=palette["KMeans"], lw=2.5)
    axes[1, 0].set_title("K-Means Elbow (Inertia ↓)", fontsize=13, weight="semibold")
    axes[1, 0].set_xlabel("k"); axes[1, 0].set_ylabel("Inertia")
    axes[1, 0].axvline(x=5, ls="--", alpha=0.4, color="red")

    # BIC / AIC (GMM only, lower = better)
    gmm_sub = metrics_df[metrics_df["algorithm"] == "GMM"]
    axes[1, 1].plot(gmm_sub["k"], gmm_sub["bic"], "o-", color="#8f2b6a", lw=2, label="BIC")
    axes[1, 1].plot(gmm_sub["k"], gmm_sub["aic"], "s--", color="#c44e9e", lw=2, label="AIC")
    axes[1, 1].set_title("GMM Information Criteria (↓)", fontsize=13, weight="semibold")
    axes[1, 1].set_xlabel("k"); axes[1, 1].set_ylabel("Score"); axes[1, 1].legend()
    axes[1, 1].axvline(x=5, ls="--", alpha=0.4, color="red")

    # Summary table in last axis
    axes[1, 2].axis("off")
    best_rows = []
    for algo in ["KMeans", "GMM", "Agglomerative"]:
        sub = metrics_df[metrics_df["algorithm"] == algo]
        best_k_sil = sub.loc[sub["silhouette"].idxmax(), "k"]
        best_k_db = sub.loc[sub["davies_bouldin"].idxmin(), "k"]
        best_k_ch = sub.loc[sub["calinski_harabasz"].idxmax(), "k"]
        best_rows.append([algo, int(best_k_sil), int(best_k_db), int(best_k_ch)])
    table = axes[1, 2].table(
        cellText=best_rows,
        colLabels=["Algorithm", "Best k (Sil↑)", "Best k (DB↓)", "Best k (CH↑)"],
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.2, 1.8)
    axes[1, 2].set_title("Optimal k Summary", fontsize=13, weight="semibold", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(fig_dir, "cluster_metric_sweep.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved metric sweep plot -> {path}")


# ─────────────────────────────────────────────────────────────────────
# 3. Bootstrap Stability Analysis (ARI)
# ─────────────────────────────────────────────────────────────────────

def bootstrap_stability(X: np.ndarray, k: int, n_bootstrap: int = 100,
                        subsample_frac: float = 0.80, random_state: int = 42) -> dict:
    """
    Assess cluster stability via bootstrap resampling.
    For each iteration: subsample 80% of the data, fit K-Means with k clusters,
    compare labels on the overlap with the full-data reference solution via ARI.
    Also test stability across 20 different random seeds on full data.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    sub_n = int(n * subsample_frac)

    # Reference solution on full data
    ref_km = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=random_state)
    ref_labels = ref_km.fit_predict(X)

    # Bootstrap ARI
    ari_scores = []
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=sub_n, replace=False)
        X_sub = X[idx]
        km_b = KMeans(n_clusters=k, n_init=10, max_iter=300,
                      random_state=int(rng.integers(0, 1_000_000)))
        labels_b = km_b.fit_predict(X_sub)
        # Re-predict reference labels on subsample using reference model
        ref_labels_sub = ref_km.predict(X_sub)
        ari = adjusted_rand_score(ref_labels_sub, labels_b)
        ari_scores.append(ari)

    # Seed sensitivity: 20 different initializations on full data
    seed_ari = []
    for s in range(20):
        km_s = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=s * 7 + 3)
        labels_s = km_s.fit_predict(X)
        seed_ari.append(adjusted_rand_score(ref_labels, labels_s))

    return {
        "bootstrap_ari_mean": np.mean(ari_scores),
        "bootstrap_ari_std": np.std(ari_scores),
        "bootstrap_ari_min": np.min(ari_scores),
        "bootstrap_ari_median": np.median(ari_scores),
        "bootstrap_ari_scores": ari_scores,
        "seed_ari_mean": np.mean(seed_ari),
        "seed_ari_std": np.std(seed_ari),
        "seed_ari_scores": seed_ari
    }


def plot_stability(stability: dict, k: int, fig_dir: str = "reports/figures"):
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Cluster Stability Analysis (k={k})", fontsize=15, weight="bold")

    # Bootstrap ARI distribution
    axes[0].hist(stability["bootstrap_ari_scores"], bins=25, color="#2b5c8f",
                 edgecolor="white", alpha=0.85)
    axes[0].axvline(stability["bootstrap_ari_mean"], color="red", ls="--", lw=2,
                    label=f"Mean ARI = {stability['bootstrap_ari_mean']:.3f}")
    axes[0].axvline(0.75, color="green", ls=":", lw=2, label="Threshold = 0.75")
    axes[0].set_title("Bootstrap Resampling ARI (B=100, 80% subsample)", fontsize=12, weight="semibold")
    axes[0].set_xlabel("Adjusted Rand Index"); axes[0].set_ylabel("Frequency")
    axes[0].legend(fontsize=10)

    # Seed sensitivity ARI
    axes[1].bar(range(len(stability["seed_ari_scores"])), stability["seed_ari_scores"],
                color="#2b8f5c", edgecolor="white", alpha=0.85)
    axes[1].axhline(stability["seed_ari_mean"], color="red", ls="--", lw=2,
                    label=f"Mean ARI = {stability['seed_ari_mean']:.3f}")
    axes[1].axhline(0.75, color="green", ls=":", lw=2, label="Threshold = 0.75")
    axes[1].set_title("Seed Sensitivity ARI (20 Initializations)", fontsize=12, weight="semibold")
    axes[1].set_xlabel("Seed Index"); axes[1].set_ylabel("ARI")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(fig_dir, "cluster_stability.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved stability plot -> {path}")


# ─────────────────────────────────────────────────────────────────────
# 4. PCA & UMAP Visualisation
# ─────────────────────────────────────────────────────────────────────

def plot_pca_umap(X: np.ndarray, labels: np.ndarray, latent_labels: np.ndarray,
                  fig_dir: str = "reports/figures"):
    os.makedirs(fig_dir, exist_ok=True)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_

    n_plots = 4 if HAS_UMAP else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    fig.suptitle("Segment Visualisation: Discovered Clusters vs. Latent Ground Truth",
                 fontsize=16, weight="bold", y=1.02)

    # PCA - discovered
    scatter0 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="Set1",
                               s=8, alpha=0.6)
    axes[0].set_title(f"PCA — Discovered Segments\n(PC1: {var_explained[0]:.1%}, PC2: {var_explained[1]:.1%})",
                      fontsize=12, weight="semibold")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    plt.colorbar(scatter0, ax=axes[0], label="Cluster")

    # PCA - latent ground truth
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    latent_enc = le.fit_transform(latent_labels)
    scatter1 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=latent_enc, cmap="Set2",
                               s=8, alpha=0.6)
    axes[1].set_title("PCA — Latent Archetypes (Ground Truth)", fontsize=12, weight="semibold")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    cbar1 = plt.colorbar(scatter1, ax=axes[1], label="Archetype")
    cbar1.set_ticks(range(len(le.classes_)))
    cbar1.set_ticklabels([c[:18] for c in le.classes_], fontsize=8)

    # UMAP
    if HAS_UMAP:
        reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
        X_umap = reducer.fit_transform(X)

        scatter2 = axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap="Set1",
                                   s=8, alpha=0.6)
        axes[2].set_title("UMAP — Discovered Segments", fontsize=12, weight="semibold")
        axes[2].set_xlabel("UMAP1"); axes[2].set_ylabel("UMAP2")
        plt.colorbar(scatter2, ax=axes[2], label="Cluster")

        scatter3 = axes[3].scatter(X_umap[:, 0], X_umap[:, 1], c=latent_enc, cmap="Set2",
                                   s=8, alpha=0.6)
        axes[3].set_title("UMAP — Latent Archetypes (Ground Truth)", fontsize=12, weight="semibold")
        axes[3].set_xlabel("UMAP1"); axes[3].set_ylabel("UMAP2")
        cbar3 = plt.colorbar(scatter3, ax=axes[3], label="Archetype")
        cbar3.set_ticks(range(len(le.classes_)))
        cbar3.set_ticklabels([c[:18] for c in le.classes_], fontsize=8)

    plt.tight_layout()
    path = os.path.join(fig_dir, "pca_umap_clusters.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved PCA/UMAP plot -> {path}")

    return var_explained


# ─────────────────────────────────────────────────────────────────────
# 5. Final Model Fitting & Persistence
# ─────────────────────────────────────────────────────────────────────

def fit_final_model(X: np.ndarray, k: int = 5, random_state: int = 42):
    """Fit the final K-Means model and return labels + model object."""
    model = KMeans(n_clusters=k, n_init=30, max_iter=1000, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels


# ─────────────────────────────────────────────────────────────────────
# 6. Main Orchestrator
# ─────────────────────────────────────────────────────────────────────

def run_segmentation(
    scaled_csv: str = "data/processed/rural_micro_entrepreneurs_scaled_features.csv",
    raw_csv: str = "data/raw/rural_micro_entrepreneurs_synthetic.csv",
    model_dir: str = "models",
    fig_dir: str = "reports/figures",
    final_k: int = 5
):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Load data
    df_scaled = pd.read_csv(scaled_csv)
    df_raw = pd.read_csv(raw_csv)
    feature_cols = [c for c in df_scaled.columns if c != "entrepreneur_id"]
    X = df_scaled[feature_cols].values
    latent_labels = df_raw["latent_archetype_label"].values

    print("=" * 65)
    print("PHASE 4: SEGMENTATION MODELLING")
    print("=" * 65)

    # Step 1: Metric sweep
    print("\n[Step 1] Evaluating k=2..10 across K-Means, GMM, Agglomerative ...")
    metrics_df = evaluate_clustering_metrics(X, k_range=range(2, 11))
    metrics_df.to_csv(os.path.join(model_dir, "clustering_metrics.csv"), index=False)
    print("  Metrics saved -> models/clustering_metrics.csv")

    # Print summary table
    print("\n  +-------------------------------------------------------------+")
    print("  |  SILHOUETTE SCORES AT k=5 (chosen k)                       |")
    print("  +-------------------------------------------------------------+")
    for algo in ["KMeans", "GMM", "Agglomerative"]:
        row = metrics_df[(metrics_df["algorithm"] == algo) & (metrics_df["k"] == final_k)]
        sil = row["silhouette"].values[0]
        db = row["davies_bouldin"].values[0]
        ch = row["calinski_harabasz"].values[0]
        print(f"  |  {algo:15s}  Sil={sil:.4f}  DB={db:.4f}  CH={ch:.1f}")
    print("  +-------------------------------------------------------------+")

    # Step 2: Plot metrics
    print("\n[Step 2] Generating metric sweep plots ...")
    plot_metric_sweep(metrics_df, fig_dir)

    # Step 3: Fit final model
    print(f"\n[Step 3] Fitting final K-Means model (k={final_k}) ...")
    final_model, final_labels = fit_final_model(X, k=final_k)
    joblib.dump(final_model, os.path.join(model_dir, "final_kmeans_model.joblib"))
    print(f"  Final model saved -> models/final_kmeans_model.joblib")

    # ARI against latent generator ground truth (internal sanity check)
    ari_vs_latent = adjusted_rand_score(latent_labels, final_labels)
    print(f"\n  Synthetic Generator Recovery Check (ARI vs. Generative Mixture Labels) = {ari_vs_latent:.4f}")
    print("  [NOTE] Confirms unsupervised K-Means successfully recovers the synthetic generator's")
    print("  underlying mixture components. This is an internal algorithmic sanity check,")
    print("  not a substitute for prospective validation on real-world MFI/RRB administrative data.")

    # Cluster sizes
    unique, counts = np.unique(final_labels, return_counts=True)
    print("\n  Cluster Size Distribution:")
    for u, c in zip(unique, counts):
        print(f"    Cluster {u}: {c:>5d} ({c/len(final_labels):.1%})")

    # Step 4: Bootstrap stability
    print(f"\n[Step 4] Bootstrap stability analysis (B=100, 80% subsample) ...")
    stability = bootstrap_stability(X, k=final_k, n_bootstrap=100)
    print(f"  Bootstrap ARI: Mean={stability['bootstrap_ari_mean']:.4f} ± {stability['bootstrap_ari_std']:.4f}")
    print(f"  Bootstrap ARI: Median={stability['bootstrap_ari_median']:.4f}, Min={stability['bootstrap_ari_min']:.4f}")
    print(f"  Seed ARI:      Mean={stability['seed_ari_mean']:.4f} ± {stability['seed_ari_std']:.4f}")
    plot_stability(stability, k=final_k, fig_dir=fig_dir)

    # Step 5: PCA & UMAP visualization
    print("\n[Step 5] PCA & UMAP visualization ...")
    var_explained = plot_pca_umap(X, final_labels, latent_labels, fig_dir)
    print(f"  PCA variance explained: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}")

    # Save labels back to engineered dataset
    df_eng = pd.read_csv("data/processed/rural_micro_entrepreneurs_engineered.csv")
    df_eng["segment_label"] = final_labels
    df_eng.to_csv("data/processed/rural_micro_entrepreneurs_segmented.csv", index=False)
    print(f"\n  Segmented dataset saved -> data/processed/rural_micro_entrepreneurs_segmented.csv")

    print("\n" + "=" * 65)
    print("PHASE 4 COMPLETE")
    print("=" * 65)

    return final_model, final_labels, metrics_df, stability


if __name__ == "__main__":
    model, labels, metrics, stab = run_segmentation()

# Clustering Algorithms

A Jupyter notebook project exploring clustering methods on rental-listing data. It implements K-means and DBSCAN in Python/NumPy, compares them with scikit-learn implementations, and evaluates Agglomerative Clustering and Gaussian Mixture Models. The notebook includes data preparation, cluster visualizations, silhouette scores, and an experiment that adds cluster labels to a Lasso regression model.

This repository is an educational notebook, not an installable Python package or command-line application.

## Project contents

| Path | Description |
| --- | --- |
| [`cluster.ipynb`](cluster.ipynb) | Main notebook containing the implementations, experiments, visualizations, and comparisons. |
| [`data/data.json`](data/data.json) | Rental-listing dataset used by the notebook. |
| [`requirements.txt`](requirements.txt) | Python dependencies for running the notebook. |

## Algorithms and experiments

- **K-means:** A from-scratch implementation and a comparison with scikit-learn's `KMeans`.
- **DBSCAN:** A from-scratch implementation that uses a KD-tree for neighborhood lookups, compared with scikit-learn's `DBSCAN`.
- **Agglomerative Clustering:** Hierarchical clustering using scikit-learn.
- **Gaussian Mixture Model:** Probabilistic clustering using scikit-learn's `GaussianMixture`.
- **Evaluation:** Cluster visualizations and silhouette scores, plus a Lasso regression experiment that measures the effect of including cluster labels as a feature.

The experiments use different feature sets, including listing coordinates and combinations of bedrooms, bathrooms, and interest level. The notebook contains example parameters; results depend on the selected features, data, and runtime environment.

## Requirements

- Python 3.9 or newer
- pip

## Setup

Run these commands from the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

Start JupyterLab from the repository root so the notebook can find `data/data.json` at its relative path:

```bash
jupyter lab cluster.ipynb
```

Alternatively, open `cluster.ipynb` in VS Code with the Jupyter extension installed. Select the Python environment where you installed the dependencies, then run the notebook from top to bottom using **Run All**. Running cells in order is important because later experiments depend on variables, preprocessing, and model definitions created earlier.

To explore a different experiment, edit the relevant notebook cells. For example, adjust `n_clusters` for K-means or Agglomerative Clustering, and `eps` or `min_samples` for DBSCAN. DBSCAN's `eps` is applied to the feature space used in the corresponding experiment, so its useful value depends on feature scale.

## Dataset and workflow

The notebook loads the included JSON data with pandas, selects rental listing fields such as price, bedrooms, bathrooms, coordinates, listing features, and interest level, and prepares features for clustering and regression. It also compares clustering approaches on geographic coordinates and on listing attributes.

The notebook includes previously saved outputs for reference. Timings and scores may differ when rerunning it because of hardware, library versions, random initialization, or changes to the data and parameters.

## License

This project is released under the [MIT License](LICENSE).

# FieldTrim :  Ablating Livestock Classification Model for Offline On-The-Field Use 


## Repository 

<https://github.com/ItamarAharony/FieldTrim>

## Overview
FieldTrim is a PyTorch-based deep learning pipeline designed to train a universal livestock mega-classifier (Chickens, Cows, Horses, and Sheep) and compresses it for edge-device deployment. The project utilizes **Singular Value Decomposition (SVD)** to factorize heavy 3x3 convolutional layers and gradually simplifies the model while retraining it between compression steps to prevent the catastrophic spatial degradation typically associated with sudden low-rank weight truncation.

This project has been developed as a final project for Course 00460211, The Technion - Israel Institute of Technology.

## Core Features
* **Dynamic Data Aggregation:** Automatically downloads, standardizes, and merges multiple disparate Kaggle datasets into a unified 32-class dataset.
* **Selective Transfer Learning:** Freezes early ResNet-18 edge-detection layers (`conv1`, `layer1`, `layer2`) while training the deep semantic layers using Kornia GPU-accelerated dynamic batch augmentation.
* **Global SVD Compression:** Flattens high-parameter 4D weight tensors and truncates their singular values based on a dynamic bottleneck `keep_ratio`, physically replacing massive convolutions with mathematically equivalent, lightweight sequential sub-layers.
* **Iterative Annealing:** Steps down the compression ratio gradually (e.g., 90%, 80%, 70%) while applying short, low-learning-rate (1e-4) recovery epochs. This prevents the cascading error of sudden ablation and allows the remaining weights to adapt.
* **Interpretability Proofs:** Automatically generates Grad-CAM spatial heatmaps and intermediate layer activation grids to visually prove that the compressed model retains the features of the animal itself rather than unintended influence from artifacts in the background.

## Prerequisites
Ensure you are running in a CUDA-enabled environment (like Google Colab) with the following libraries installed:
* `torch` & `torchvision`
* `kornia` (for GPU-accelerated augmentations)
* `thop` (for FLOPs profiling)
* `kagglehub` (for dataset retrieval)
* `opencv-python` (for Grad-CAM heatmap generation)
* `scikit-learn` & `matplotlib`

## Usage
The entire data pipeline is run by calling the master function: `process_mega_dataset`.

To execute the full training, ablation, and visualization sequence, define your datasets and trigger the pipeline as follows:

```python
# 1. Define the constituent datasets
livestock_datasets = [
    {
        "name": "Chickens",
        "url": "galibusm/galliformespectra-a-hen-breed-dataset",
        "subfolder": "train(Original)",
        "type": "standard"
    },
    {
        "name": "Cows",
        "url": "zaidworks0508/cow-breed-classification-dataset",
        "subfolder": "Cow Breed Dataset",
        "type": "standard"
    },
    {
        "name": "Sheep",
        "url": "divyansh22/sheep-breed-classification",
        "subfolder":"SheepFaceImages",
        "type":"standard"
    },
    {
        "name": "Horses",
        "url": "olgabelitskaya/horse-breeds",
        "subfolder": None,
        "type": "horse"
    }
]

# 2. Execute the Master Pipeline
process_mega_dataset(
    datasets_info=livestock_datasets,
    experiment_name="Mega_Livestock",
    num_epochs=10,
    keep_ratios=[0.9**i for i in range(10)], # Generates the annealing curve
    recovery_epochs=3, # amount of training epochs after each ablation, meant to retrain the network after its structure has been changed
    subsample_fraction=1.0,
    layer_target_obs="layer4[1].conv2" # Target layer for Grad-CAM & Feature Maps
)

```

## Pipeline Architecture

1. **Data Engineering (`build_universal_dataset`):** Intercepts Kaggle downloads, dynamically parses prefixes (for datasets like horses), cleans junk classes (e.g., "unidentified (mixed)"), and performs stratified sub-sampling.
2. **Baseline Training:** Standard cross-entropy optimization to establish the initial, uncompressed weights and measure raw transfer-learning accuracy.
3. **The Annealing Loop:** The network sweeps through the provided `keep_ratios`. At each ratio step, SVD sequential blocks are mathematically injected, FLOPs are profiled, the immediate "sudden drop" damage is logged, and recovery epochs are run to heal the parameter matrices.
4. **Visualization:** Generates three core output figures:
* Global Accuracy vs. Compression Ratio (Trade-off Curve).
* Grad-CAM focus overlays on a test subject.
* Side-by-side feature map integrity grids (Baseline vs. Sudden Compression vs. Gradual Compression (Annealing)).

# Indian Sign Language (ISL) Alphabet Classification

A Jupyter Notebook project for image classification using an Indian Sign
Language (ISL) alphabet dataset. It adapts the earlier Fashion-MNIST
CNN/DNN workflow to alphabet-labelled image folders.

## Project objective

Train and evaluate image-classification models that predict the alphabet
represented by a hand-sign image. The experiments also investigate how
translating (shifting) an image affects model accuracy.

## Dataset structure

Each alphabet folder is treated as one class:

``` text
project/
├── ISL_CNN_Experiments.ipynb
├── ISL_Dataset.zip              # optional
└── ISL_Dataset/
    ├── A/
    ├── B/
    ├── C/
    └── ...
```

The supplied ZIP inspected for this project contained **702 images in 23
alphabet folders**:

`A, B, C, D, E, F, G, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Z`

The folders **H, J, and Y** were not present in that ZIP. The notebook
detects the folders that actually exist and does not create missing
classes. Add the missing folders and rerun the notebook if you later
obtain those images.

The supplied sample image was 128 × 128 RGB JPEG. The notebook inspects
the full dataset and resizes readable images to 64 × 64 RGB for the
initial experiments.

## Notebook contents

Main file: **`ISL_CNN_Experiments.ipynb`**

The notebook includes:

1.  Library imports, random seeds, and environment checks.
2.  Dataset path detection and optional ZIP extraction.
3.  Folder/class discovery, image counts, file formats, dimensions, and
    unreadable-file checks.
4.  Visualization of examples from alphabet folders.
5.  Image resizing, RGB conversion, normalization, and label encoding.
6.  Stratified train, validation, and test splits.
7.  Class-distribution analysis and class weights.
8.  Scikit-learn MLP baseline.
9.  NumPy circular-shift robustness experiment for the MLP.
10. Convolution output-size calculations.
11. A 2D convolution implementation using NumPy.
12. Blur, sharpen, Sobel X, Sobel Y, and edge-magnitude demonstrations.
13. CNN construction, training, and evaluation.
14. CNN training/validation accuracy and loss plots.
15. CNN confusion matrix and classification report.
16. DNN construction, training, and evaluation.
17. MLP/CNN/DNN test-accuracy comparison.
18. OpenCV zero-filled image translations.
19. CNN and DNN evaluation on translated test images.
20. Accuracy-drop calculations, per-class analysis, sample prediction,
    and model saving.

## Requirements

Install the main dependencies:

``` bash
pip install numpy pandas matplotlib pillow scikit-learn tensorflow opencv-python jupyter
```

A GPU is optional. Training can run on CPU but may take longer.

## How to run

### Option 1: Extract the dataset

1.  Place `ISL_CNN_Experiments.ipynb` in your project directory.

2.  Extract the dataset so that `ISL_Dataset/` is beside the notebook.

3.  Open a terminal in that directory and run:

    ``` bash
    jupyter notebook
    ```

4.  Open `ISL_CNN_Experiments.ipynb`.

5.  Run all cells from top to bottom.

### Option 2: Keep the ZIP beside the notebook

Place `ISL_Dataset.zip` beside the notebook. The dataset-loading cell
attempts to extract it automatically if the `ISL_Dataset` folder is not
found.

### Google Colab

Upload the ZIP to the Colab session or mount Google Drive. If the
dataset is stored elsewhere, update `DATASET_DIR` in the dataset-loading
cell to point to the extracted folder.

## Preprocessing

-   Image mode: RGB (3 channels).
-   Target image size: 64 × 64 pixels.
-   Pixel scaling: divide by 255, producing values from 0 to 1.
-   Labels: folder names mapped to integer class indices.
-   Data splitting: stratified training, validation, and test sets.
-   Training augmentation: small translations, rotations, zoom, and
    contrast changes in the CNN training pipeline only.

The notebook uses the classes discovered in the dataset. The output
layer therefore has one neuron per detected class: 23 for the supplied
ZIP, or 26 if all A--Z folders are present.

## Models

### MLP baseline

A Scikit-learn `MLPClassifier` is trained on flattened RGB image arrays.
It provides a fully connected baseline but does not explicitly preserve
the spatial layout of image pixels.

### CNN

The Keras CNN uses convolution and max-pooling layers to learn local
image features, followed by dense layers for alphabet classification.
Its softmax output size matches the number of detected classes.

### DNN

The Keras DNN flattens each image and uses fully connected dense layers.
It provides a comparison against the CNN architecture.

## Image-shift robustness experiment

The notebook evaluates trained models on translated test images.

-   NumPy `roll` shifts are circular: pixels leaving one edge wrap
    around to the opposite edge.
-   OpenCV translations fill newly exposed pixels with a constant black
    border.
-   Models are evaluated without retraining on shifted images.
-   Original labels are retained because translation does not change the
    intended alphabet.

Accuracy drop is reported in percentage points:

``` text
Accuracy drop = original accuracy (%) - shifted accuracy (%)
```

A positive value means accuracy decreased after shifting.

## Generated outputs

The notebook saves files in `isl_model_outputs/`:

-   `isl_cnn.keras` --- trained CNN model.
-   `isl_dnn.keras` --- trained DNN model.
-   `class_mapping.json` --- class/index mapping and input image
    settings.
-   `shift_robustness_results.csv` --- CNN/DNN shift evaluation results.
-   `model_accuracy_comparison.csv` --- original test accuracy
    comparison.

## Important limitations

-   The supplied dataset has 23 classes, not all 26 English alphabet
    folders. H, J, and Y were missing from the supplied ZIP.
-   The dataset is small and class counts are uneven. Test metrics can
    vary with the random split and should be interpreted cautiously.
-   Static image classification cannot learn the temporal movement of
    signs that require motion. Video sequences and a temporal model such
    as CNN-LSTM would be needed for dynamic signs.
-   Translation tests cover simple shifts only. They do not fully
    represent changes in rotation, scale, lighting, viewpoint, camera
    distance, or background.
-   If near-duplicate images or images from the same signer/session
    appear in both training and test sets, evaluation may be overly
    optimistic. A signer/session-independent split is preferable when
    metadata is available.
-   High accuracy on this dataset alone does not establish real-world
    sign-language recognition performance.

## Suggested future improvements

1.  Add the missing alphabet folders and collect more images per class.
2.  Collect images from multiple signers, backgrounds, lighting
    conditions, and camera viewpoints.
3.  Use signer-independent train/validation/test splits where possible.
4.  Compare RGB and grayscale preprocessing.
5.  Test transfer learning with a pretrained model such as MobileNetV2.
6.  Evaluate rotation, scale, brightness, blur, and background
    robustness.
7.  For dynamic signs, collect video sequences and investigate CNN-LSTM
    or other temporal models.

## Reproducibility

The notebook sets Python, NumPy, and TensorFlow random seeds to `42`.
Exact results can still vary across hardware, TensorFlow versions, and
nondeterministic GPU operations.

------------------------------------------------------------------------

**Project note:** This is an educational computer-vision experiment and
a starting point for ISL alphabet classification. It is not a validated
assistive communication system.

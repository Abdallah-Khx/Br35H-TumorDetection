# Brain Tumor Detection using Mask R-CNN (ResNet50 + FPN)

## Overview
This project focuses on automating brain tumor detection from MRI scans using advanced deep learning techniques. The goal is to build a fast, reliable, and clinically applicable AI tool that assists radiologists in early diagnosis, reducing delays and improving patient outcomes — especially in underserved areas with limited access to specialists.

## Dataset

**Dataset:** [Br35H Brain Tumor Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)

- Contains labeled **MRI scans** with tumor presence or absence.

- Includes `.jpg` images and `annotations.json` files with segmentation masks.

- Data was converted to **COCO format** for model training and evaluation.

**Preprocessing Steps:**

- Normalized pixel values to ensure consistent input.

- Preserved original image dimensions to avoid losing spatial details.

- Applied data augmentation (rotations, flips, etc.) to reduce overfitting.

## Model Architecture
**Mask R-CNN with ResNet50 + FPN Backbone**

- **ResNet50** acts as a powerful feature extractor, identifying textures, edges, and shapes.

- **Feature Pyramid Network (FPN)** enhances multi-scale feature detection, allowing accurate recognition of both small and large tumors.

- The **Mask R-CNN** performs:

  - **Classification –** identifies tumor vs. background.

  - **Localization –** draws bounding boxes around detected regions using **Open-Cv**.

  - **Segmentation –** produces precise pixel-level tumor masks.


## Training Methodology
**Model:** Mask R-CNN with ResNet50 backbone

**Feature Extractor:** Feature Pyramid Network (FPN)

**Segmentation Format:** COCO JSON

**Batch Size:** 4

**Optimizer:** Adam

**Learning Rate:** 0.0005

**Epochs:** 10

## Results & Insights

The Mask R-CNN model achieved strong detection and<br> 
segmentation performance across multiple evaluation metrics.

| Train | Precision | F1-score | Recall | AUC | Test |
|:------:|:-----------:|:----------:|:--------:|:------:|:------:|
| **96.47%** | **98.62%** | **94.67%** | **96%** | **91.7%** | **97.80%** |

**Key Observations**:
- The model achieved a high training precision of **98.62%** and an F1-score of **94.67%**, indicating balanced accuracy between detection and segmentation.  
- A test performance of **97.80%** demonstrates strong generalization capability.  
- The AUC of **91.7%** confirms the model’s reliable discriminative ability between tumor and non-tumor regions.

## Model Predictions

Below are sample MRI scans processed by the trained **Mask R-CNN model**.<br>
<br>
Each prediction shows the detected tumor region highlighted with a bounding box using **Open-Cv**, demonstrating accurate localization across multiple brain tumor cases.

| **Sample 1** | **Sample 2** | **Sample 3** | **Sample 4** |
|:---------:|:---------:|:---------:|:---------:|
| <img width="267" height="289" alt="image" src="https://github.com/user-attachments/assets/30dd0357-0f9d-4d0c-a656-b064e71cba29" /> | <img width="248" height="289" alt="image" src="https://github.com/user-attachments/assets/41581f59-3951-41ad-b974-223210ec9c8d" /> | <img width="272" height="289" alt="image" src="https://github.com/user-attachments/assets/686bf9eb-bb3c-471e-b1da-2e884a600e7d" /> | <img width="256" height="286" alt="image" src="https://github.com/user-attachments/assets/fced2b30-8881-449e-b1d1-2310687dd7ef" /> |

Each output demonstrates the model’s capability to detect and segment tumor regions precisely, confirming strong predictive performance suitable for clinical integration.

## Future Work

- Expand dataset for multi-class tumor classification (benign vs malignant).

- Integrate explainable AI (XAI) features for transparency.

- Deploy via cloud services (Azure / AWS) for scalable medical use.

- Collaborate with healthcare providers for clinical validation.

## Contributors

**Authors**: *Abdallah Khader, Khalid  Naif, Faris Wazni*

**Affiliation**: *Bahçeşehir University — AI/CS Engineering Capstone Project*

**Year**: *2025*








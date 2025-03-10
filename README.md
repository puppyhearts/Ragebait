# Ragebait Classification Project

This repository contains the codebase for **Ragebait Detection and Classification of Sentiment Manipulation through Social Media**, which uses retrofitted GloVe embeddings, SBERT, and cosine similarity to classify text descriptions of images as *ragebait* or *not ragebait*. The repository processes images, generates descriptions, evaluates various classification methods, and logs results.

---

## Repository Structure

### **Core Files**

1. **`create_descriptions.py`**
   - **Purpose**: Converts images into textual descriptions using ChatGPT 40-mini
   - **How to Use**:
     - Place images in the appropriate folder (e.g., `images`).
     - Run the script to generate descriptions in the `descriptions` folder.

2. **`demo_newimage.py`**
   - **Purpose**: Demonstrates the classification pipeline for a single image.
   - **How to Use**:
     - Run the script to generate a description, classify it, and explain why it is or isn’t ragebait.
     - Note: This description isn't saved and won't contribute to retrofitting unless it is named appropriately and pplaced in the images folder

3. **`Evaluation_3methods.py`**
   - **Purpose**: Compares three classification approaches:
     - GloVe-based cosine similarity.
     - SBERT embeddings.
     - Retrofitted GloVe embeddings.
     - Generates Precision, Recall, F1-score and accuracy metrics
   - **How to Use**:
     - Run the script to evaluate the performance of the 3 methods on classifying the dataset and generate a detailed performance log.

4. **`logFile.txt`**
   - **Purpose**: Stores the results of classification (predicted vs. true labels) for analysis.
   - **How to Use**:
     - Refer to this file after running the evaluation scripts for detailed metrics and comparisons.

5. **`novel_tester.py`**
   - **Purpose**: A utility script for testing modifications to the retrofitting algorithm or embeddings on a subset of the dataset.
   - **How to Use**:
     - Run to validate new experimental methods or parameters.

6. **`retrofit_reevaluate.py`**
   - **Purpose**: Implements the primary classification logic to retrofit GloVe embeddings.
   - **How to Use**:
     - Run this script after generating descriptions to classify the dataset and retrofit embeddings that can classify ragebait.

---

## Additional Resources not present

- **`glove.840B.300d.txt`**:
  - Pre-trained GloVe embeddings used for baseline classification.

- **`glove_retrofitted.txt`**:
  - Retrofitted GloVe embeddings incorporating domain-specific modifications.

- **Folders**:
  - **`MAMI_test`**: Contains input images from the MAMI dataset.
  - **`MAMI_summary`**: Contains text descriptions of the images and the `test_labels.txt` file with ground truth labels - generated by create_descriptions.py.
  - **`images`**: Custom dataset with 800+ images. What I've left in here is a toy dataset with 10 entries of each type
  - **`ddescriptions`**: Custom dataset with 800+ descriptions of the images in images described by GPT 4o. What I've left in here is a toy dataset with 10 entries of each type.

---

## Workflow

1. **Generate Descriptions**:
   - Use `create_descriptions.py` to create text descriptions for images in the `images` folder.

2. **Classify Descriptions**:
   - Use `retrofit.py` to retrofit GloVe and classify the descriptions using retrofitted GloVe embeddings.
   - Alternatively, run `Evaluation_3methods.py` to only compare GloVe, SBERT, and retrofitted GloVe methods.

3. **Evaluate Results**:
   - Use `logFile.txt` to review performance metrics, including precision, recall, F1-score, and accuracy.

4. **Demo New Images**:
   - Use `demo_newimage.py` to classify and analyze a single image through the pipeline.

---

## Requirements

- Python 3.8 or higher
- Install dependencies with `pip install -r requirements.txt`

---

## Future Work

- Incorporate larger datasets for better generalization.
- Experiment with building an LLM to retrofit better using LLAMA or other transformer models for improved accuracy.
- Develop a lightweight GUI for real-time image classification.

---

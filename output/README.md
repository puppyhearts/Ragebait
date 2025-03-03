# Ragebait Classification Project
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

## Additional Resources not present

- **`glove.840B.300d.txt`**:
  - Pre-trained GloVe embeddings used for baseline classification.

- **`glove_retrofitted.txt`**:
  - Retrofitted GloVe embeddings incorporating domain-specific modifications.

3. **Evaluate Results**:
   - Run `Evaluation_3methods.py` to review performance metrics, including precision, recall, F1-score, and accuracy on the toy dataset present in the images and descriptions folders.

4. **Demo New Images**:
   - Use `demo_newimage.py` to classify and analyze a single image through the pipeline.

---

## Requirements

- Python 3.8 or higher
- Install dependencies with `pip install -r requirements.txt`

---


---

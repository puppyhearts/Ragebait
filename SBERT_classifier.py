import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sentence_transformers import SentenceTransformer

# Load SBERT model
def load_sbert_model():
    """
    Load the SBERT model.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")  # You can replace with another model if needed

# Vectorize text using SBERT
def vectorize_text_with_sbert(text, model):
    """
    Vectorize the given text using SBERT.
    """
    return model.encode(text, convert_to_tensor=True)

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    """
    # Move tensors to CPU and convert to NumPy
    vec1 = vec1.cpu().numpy()
    vec2 = vec2.cpu().numpy()
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Classify description using SBERT
def classify_description_with_sbert(description, model):
    """
    Classify if a description is 'ragebait' or 'not ragebait' using SBERT.
    """
    # Define reference embeddings
    ragebait_ref = "misogyny, violence against women, hating women, Sexist, Patriarchy, Objectification, Victimization, Hypersexualization, Gender violence"
    non_ragebait_ref = "not misogynistic"

    ragebait_vec = vectorize_text_with_sbert(ragebait_ref, model)
    non_ragebait_vec = vectorize_text_with_sbert(non_ragebait_ref, model)

    desc_vec = vectorize_text_with_sbert(description, model)

    # Calculate cosine similarity
    similarity_to_ragebait = cosine_similarity(desc_vec, ragebait_vec)
    similarity_to_non_ragebait = cosine_similarity(desc_vec, non_ragebait_vec)

    # Determine classification
    return "ragebait" if similarity_to_ragebait > similarity_to_non_ragebait else "not ragebait"

# Accuracy checker
def accuracy_checker_sbert(descriptions_folder, model):
    """
    Check accuracy by comparing file names and predictions, outputting metrics.
    """
    y_true = []
    y_pred = []

    print("Processing descriptions...")
    for filename in os.listdir(descriptions_folder):
        if filename.endswith("_description.txt"):
            # Determine true label based on file naming convention
            if "ENG" in filename and filename.replace("res_ENG", "").replace("_description.txt", "").isdigit():
                true_label = "ragebait"
            elif "ENGN" in filename and filename.replace("res_ENGN", "").replace("_description.txt", "").isdigit():
                true_label = "not ragebait"
            else:
                continue  # Skip files not following the convention

            # Load description
            with open(os.path.join(descriptions_folder, filename), "r", encoding="utf-8") as file:
                description = file.read().strip()

            # Predict label
            predicted_label = classify_description_with_sbert(description, model)

            # Store results
            y_true.append(true_label)
            y_pred.append(predicted_label)
            print(f"File: {filename} | Predicted: {predicted_label} | True: {true_label}")

    # Calculate metrics
    if y_true and y_pred:
        precision = precision_score(y_true, y_pred, pos_label="ragebait", average="binary")
        recall = recall_score(y_true, y_pred, pos_label="ragebait", average="binary")
        f1 = f1_score(y_true, y_pred, pos_label="ragebait", average="binary")
        accuracy = accuracy_score(y_true, y_pred)
        num_classified = len(y_true)

        print(f"\nTotal number of entries classified: {num_classified}") 
        print("\nMetrics:")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Number of classified entries: {len(y_true)}")
    else:
        print("No valid classifications made.")

# Main function
if __name__ == "__main__":
    descriptions_folder = "descriptions"  # Path to folder containing description .txt files

    print("Loading SBERT model...")
    sbert_model = load_sbert_model()

    print("Checking accuracy using SBERT...")
    accuracy_checker_sbert(descriptions_folder, sbert_model)
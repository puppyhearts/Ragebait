import os
import numpy as np
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sentence_transformers import SentenceTransformer

def load_glove_vectors(glove_path):
    glove_dict = {}
    with open(glove_path, "r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()
            if len(values) != 301:  # 1 word + 300 dimensions
                continue
            word = values[0]
            vector = np.array(values[1:], dtype="float32")
            glove_dict[word] = vector
    return glove_dict

def vectorize_text(text, glove_dict):
    words = text.split()
    vectors = [glove_dict[word] for word in words if word in glove_dict]
    return np.mean(vectors, axis=0) if vectors else None


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

def classify_description_RETRO(description, glove_dict):
    ragebait_vec = glove_dict.get("ragebait")
    non_ragebait_vec = glove_dict.get("non_ragebait")
    if ragebait_vec is None or non_ragebait_vec is None:
        print("Error: Key vectors not found in GloVe.")
        return None
    desc_vec = vectorize_text(description, glove_dict)
    if desc_vec is None:
        return None
    sim_ragebait = np.dot(desc_vec, ragebait_vec) / (np.linalg.norm(desc_vec) * np.linalg.norm(ragebait_vec))
    sim_non_ragebait = np.dot(desc_vec, non_ragebait_vec) / (np.linalg.norm(desc_vec) * np.linalg.norm(non_ragebait_vec))
    return "ragebait" if sim_ragebait > sim_non_ragebait else "not ragebait"


def classify_description_GLOVE(description, glove_dict):
    ragebait_vec = vectorize_text("misogyny, violence against women, hating women, Sexist, Patriarchy, Objectification, Victimization, Hypersexualization, Gender violence", glove_dict)
    non_ragebait_vec = vectorize_text("not misogyny", glove_dict)
    if ragebait_vec is None or non_ragebait_vec is None:
        print("Error: Key vectors not found in GloVe.")
        return None
    desc_vec = vectorize_text(description, glove_dict)
    if desc_vec is None:
        return None
    sim_ragebait = np.dot(desc_vec, ragebait_vec) / (np.linalg.norm(desc_vec) * np.linalg.norm(ragebait_vec))
    sim_non_ragebait = np.dot(desc_vec, non_ragebait_vec) / (np.linalg.norm(desc_vec) * np.linalg.norm(non_ragebait_vec))
    return "ragebait" if sim_ragebait > sim_non_ragebait else "not ragebait"

# Classify description using SBERT
def classify_description_SBERT(description, model):
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


def load_test_labels(label_file_path):
    true_labels = {}
    with open(label_file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                file_number = re.search(r'\d+', parts[0]).group(0)
                label = int(parts[1])
                true_labels[file_number] = "ragebait" if label == 1 else "not ragebait"
    return true_labels

def evaluate_model(descriptions_folder, glove_dict, label_file_path, model):
    true_labels = load_test_labels(label_file_path)
    labeled_descriptions = []
    y_true, y_pred = [], []

    for filename in os.listdir(descriptions_folder):
        if filename.endswith("_description.txt"):
            file_number = re.search(r'\d+', filename).group(0)
            if file_number in true_labels:
                true_label = true_labels[file_number]
                with open(os.path.join(descriptions_folder, filename), "r", encoding="utf-8") as file:
                    description = file.read().strip()
                labeled_descriptions.append((description, true_label))

    for description, true_label in labeled_descriptions:
        if model.endswith("GLOVE"):
            predicted_label = classify_description_GLOVE(description, glove_dict)
            if predicted_label is not None:
                y_true.append(true_label)
                y_pred.append(predicted_label)
                print(f"File: {filename} | Predicted: {predicted_label} | True: {true_label}")

    for description, true_label in labeled_descriptions:
        if model.endswith("retrofitted"):
            predicted_label = classify_description_RETRO(description, glove_dict)
            if predicted_label is not None:
                y_true.append(true_label)
                y_pred.append(predicted_label)
                print(f"File: {filename} | Predicted: {predicted_label} | True: {true_label}")

    for description, true_label in labeled_descriptions:
        if model.endswith("SBERT"):
            predicted_label = classify_description_SBERT(description, glove_dict) #Despite the variable name, I'm passing the SBERT model here
            if predicted_label is not None:
                y_true.append(true_label)
                y_pred.append(predicted_label)
                print(f"File: {filename} | Predicted: {predicted_label} | True: {true_label}")

    precision = precision_score(y_true, y_pred, pos_label="ragebait")
    recall = recall_score(y_true, y_pred, pos_label="ragebait")
    f1 = f1_score(y_true, y_pred, pos_label="ragebait")
    accuracy = accuracy_score(y_true, y_pred)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    descriptions_folder = "MAMI_summary"
    label_file_path = "MAMI_summary/test_labels.txt"

    #Method 1
    glove_path = "glove.840B.300d.txt"
    print("OG GLOVE Cosine similarity evaluation")
    glove_dict = load_glove_vectors(glove_path)
    evaluate_model(descriptions_folder, glove_dict, label_file_path, 'og_GLOVE')

    #Method 2
    glove_path = "glove_retrofitted.txt"
    print("Retrofitted GLOVE Cosine similarity evaluation")
    glove_dict = load_glove_vectors(glove_path)
    evaluate_model(descriptions_folder, glove_dict, label_file_path, 'retrofitted')

    #Method3
    glove_path = "glove_retrofitted.txt"
    print("Retrofitted GLOVE Cosine similarity evaluation")
    sbert_dict = SentenceTransformer("all-MiniLM-L6-v2")
    evaluate_model(descriptions_folder, sbert_dict, label_file_path, 'SBERT')

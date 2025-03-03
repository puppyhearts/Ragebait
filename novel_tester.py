import os
import numpy as np
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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

def save_glove_vectors(glove_dict, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        for word, vector in glove_dict.items():
            vector_str = " ".join(map(str, vector))
            file.write(f"{word} {vector_str}\n")

def vectorize_text(text, glove_dict):
    words = text.split()
    vectors = [glove_dict[word] for word in words if word in glove_dict]
    return np.mean(vectors, axis=0) if vectors else None

def retrofit_glove(glove_dict, labeled_descriptions):
    ragebait_vectors = []
    non_ragebait_vectors = []

    for description, label in labeled_descriptions:
        desc_vec = vectorize_text(description, glove_dict)
        if desc_vec is not None:
            if label == "ragebait":
                ragebait_vectors.append(desc_vec)
            else:
                non_ragebait_vectors.append(desc_vec)

    if ragebait_vectors:
        glove_dict["ragebait"] = np.mean(ragebait_vectors, axis=0)
    if non_ragebait_vectors:
        glove_dict["non_ragebait"] = np.mean(non_ragebait_vectors, axis=0)


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def classify_description(description, glove_dict):
    ragebait_vec = glove_dict.get("ragebait")
    non_ragebait_vec = glove_dict.get("non_ragebait")
    if ragebait_vec is None or non_ragebait_vec is None:
        print("Error: Key vectors not found in GloVe.")
        return None
    desc_vec = vectorize_text(description, glove_dict)
    if desc_vec is None:
        return None
    sim_ragebait = cosine_similarity(desc_vec, ragebait_vec)
    sim_non_ragebait = cosine_similarity(desc_vec, non_ragebait_vec)
    return "ragebait" if sim_ragebait > sim_non_ragebait else "not ragebait"

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

def evaluate_model(descriptions_folder, glove_dict, label_file_path, output_path):
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

    retrofit_glove(glove_dict, labeled_descriptions)

    for description, true_label in labeled_descriptions:
        predicted_label = classify_description(description, glove_dict)
        if predicted_label is not None:
            y_true.append(true_label)
            y_pred.append(predicted_label)
            print(f"{filename:<40} {true_label:<15} {predicted_label:<15}")

    save_glove_vectors(glove_dict, output_path)

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
    glove_path = "glove_retrofitted.txt"

    glove_dict = load_glove_vectors(glove_path)
    evaluate_model(descriptions_folder, glove_dict, label_file_path, glove_path)
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def load_glove_vectors(glove_path):
    glove_dict = {}
    with open(glove_path, "r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, 1):
            values = line.strip().split()
            if len(values) != 301:
                continue
            word = values[0]
            try:
                vector = np.array(values[1:], dtype="float32")
                glove_dict[word] = vector
            except ValueError:
                continue
    print(f"Loaded {len(glove_dict)} GloVe vectors.")
    return glove_dict

def vectorize_text(text, glove_dict):
    words = text.lower().split()
    vectors = [glove_dict[word] for word in words if word in glove_dict]
    if not vectors:
        print(f"No vectors found for text: {text}")
    return np.mean(vectors, axis=0) if vectors else None

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def classify_description(description, glove_dict):
    ragebait_vec = vectorize_text("misogyny, violence against women, hating women, Sexist, Patriarchy, Objectification, Victimization, Hypersexualization, Gender violence", glove_dict)
    non_ragebait_vec = vectorize_text("not misogyny", glove_dict)

    if ragebait_vec is None or non_ragebait_vec is None:
        print("Key terms not found in GloVe. Cannot classify.")
        return None

    desc_vec = vectorize_text(description, glove_dict)
    if desc_vec is None:
        print(f"Description vectorization failed: {description}")
        return None

    similarity_to_ragebait = cosine_similarity(desc_vec, ragebait_vec)
    similarity_to_non_ragebait = cosine_similarity(desc_vec, non_ragebait_vec)

    print(f"Similarity to 'ragebait': {similarity_to_ragebait:.2f}, "
          f"to 'not ragebait': {similarity_to_non_ragebait:.2f}")

    #return "ragebait" if similarity_to_ragebait > similarity_to_non_ragebait else "not ragebait"
    return "ragebait" if similarity_to_ragebait > 0.51 else "not ragebait"

def accuracy_checker(descriptions_folder, glove_dict):
    y_true = []
    y_pred = []

    print("\nClassification Results:")
    print(f"{'Filename':<40} {'True Label':<15} {'Predicted Label':<15}")
    print("=" * 70)

    for filename in os.listdir(descriptions_folder):
        print(f"Processing file: {filename}")
        if filename.endswith(".txt"):
            if "res_ENG" in filename and filename.replace("res_ENG", "").replace("_description.txt", "").isdigit():
                true_label = "ragebait"
            elif "res_ENGN" in filename and filename.replace("res_ENGN", "").replace("_description.txt", "").isdigit():
                true_label = "not ragebait"
            else:
                print(f"Skipping file: {filename} (Invalid format)")
                continue

            with open(os.path.join(descriptions_folder, filename), "r", encoding="utf-8") as file:
                description = file.read().strip()

            predicted_label = classify_description(description, glove_dict)
            if predicted_label is not None:
                y_true.append(true_label)
                y_pred.append(predicted_label)
                print(f"{filename:<40} {true_label:<15} {predicted_label:<15}")

    if y_true and y_pred:
        precision = precision_score(y_true, y_pred, pos_label="ragebait", average="binary")
        recall = recall_score(y_true, y_pred, pos_label="ragebait", average="binary")
        f1 = f1_score(y_true, y_pred, pos_label="ragebait", average="binary")
        accuracy = accuracy_score(y_true, y_pred)

        # Count classified entries
        print(f"\nTotal number of entries classified: {len(y_true)}")

        print("\nMetrics:")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")
        print(f"Accuracy: {accuracy:.2f}")
    else:
        print("No valid classifications made.")

if __name__ == "__main__":
    descriptions_folder = "descriptions"
    glove_path = "glove.840B.300d.txt"

    print("Loading GloVe vectors...")
    glove_dict = load_glove_vectors(glove_path)

    print("Checking accuracy...")
    accuracy_checker(descriptions_folder, glove_dict)
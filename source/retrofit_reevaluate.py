import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def load_glove_vectors(glove_path):
    """
    Load GloVe vectors from a file into a dictionary with error handling.
    """
    glove_dict = {}
    with open(glove_path, "r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, 1):
            values = line.strip().split()
            word = values[0]
            vector_data = values[1:]
            
            if len(vector_data) != 300:  # Ensure correct format
                continue
            
            try:
                vector = np.array(vector_data, dtype="float32")
                glove_dict[word] = vector
            except ValueError:
                continue
    return glove_dict

def save_glove_vectors(glove_dict, output_path):
    """
    Save updated GloVe vectors to a file.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        for word, vector in glove_dict.items():
            vector_str = " ".join(map(str, vector))
            file.write(f"{word} {vector_str}\n")

def vectorize_text(text, glove_dict):
    """
    Vectorize a given text using GloVe vectors and return the centroid.
    """
    words = text.split()
    vectors = [glove_dict[word] for word in words if word in glove_dict]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return None

def retrofit_glove(glove_dict, labeled_descriptions):
    """
    Retrofit GloVe vectors with labeled descriptions for 'ragebait' and 'non_ragebait'.
    """
    ragebait_vectors = []
    non_ragebait_vectors = []

    for description, label in labeled_descriptions:
        desc_vec = vectorize_text(description, glove_dict)
        if desc_vec is not None:
            if label == "ragebait":
                ragebait_vectors.append(desc_vec)
            elif label == "not ragebait":
                non_ragebait_vectors.append(desc_vec)
    
    # Compute centroids for "ragebait" and "non_ragebait"
    if ragebait_vectors:
        ragebait_centroid = np.mean(ragebait_vectors, axis=0)
        glove_dict["ragebait"] = ragebait_centroid  # Add "ragebait" vector explicitly
    else:
        print("Warning: No valid 'ragebait' descriptions to compute centroid.")
        ragebait_centroid = None

    if non_ragebait_vectors:
        non_ragebait_centroid = np.mean(non_ragebait_vectors, axis=0)
        glove_dict["non_ragebait"] = non_ragebait_centroid  # Add "non_ragebait" vector explicitly
    else:
        print("Warning: No valid 'non_ragebait' descriptions to compute centroid.")
        non_ragebait_centroid = None

    # Retrofit GloVe vectors
    if ragebait_centroid is not None:
        for word in glove_dict.keys():
            word_vec = glove_dict[word]
            glove_dict[word] += 0.1 * (ragebait_centroid - word_vec)
    
    if non_ragebait_centroid is not None:
        for word in glove_dict.keys():
            word_vec = glove_dict[word]
            glove_dict[word] += 0.1 * (non_ragebait_centroid - word_vec)
    
    return glove_dict

def classify_description(description, glove_dict):
    """
    Classify a description using retrofitted GloVe vectors.
    """
    ragebait_vec = glove_dict.get("ragebait")
    non_ragebait_vec = glove_dict.get("non_ragebait")
    
    if ragebait_vec is None or non_ragebait_vec is None:
        print("Error: Key vectors ('ragebait' or 'non_ragebait') not found in GloVe.")
        return None
    
    desc_vec = vectorize_text(description, glove_dict)
    if desc_vec is None:
        return None  # Cannot classify if no valid vector

    similarity_to_ragebait = np.dot(desc_vec, ragebait_vec) / (np.linalg.norm(desc_vec) * np.linalg.norm(ragebait_vec))
    similarity_to_non_ragebait = np.dot(desc_vec, non_ragebait_vec) / (np.linalg.norm(desc_vec) * np.linalg.norm(non_ragebait_vec))
    
    return "ragebait" if similarity_to_ragebait > similarity_to_non_ragebait else "not ragebait"

def evaluate_model(descriptions_folder, glove_dict, output_path):
    """
    Evaluate the model using descriptions, retrofit GloVe, and compute metrics.
    """
    y_true = []
    y_pred = []
    labeled_descriptions = []

    # Load descriptions and determine labels
    for filename in os.listdir(descriptions_folder):
        if filename.endswith(".txt"):
            # Determine true label
            if "ENG" in filename and "ENGN" not in filename:
                true_label = "ragebait"
            elif "ENGN" in filename:
                true_label = "not ragebait"
            else:
                continue  # Skip files not following the convention
            
            # Load description
            with open(os.path.join(descriptions_folder, filename), "r", encoding="utf-8") as file:
                description = file.read().strip()
            
            # Add to labeled descriptions for retrofitting
            labeled_descriptions.append((description, true_label))
    
    # Retrofit GloVe
    glove_dict = retrofit_glove(glove_dict, labeled_descriptions)

    # Save the retrofitted GloVe vectors
    save_glove_vectors(glove_dict, output_path)

    # Classify descriptions using retrofitted GloVe
    for description, true_label in labeled_descriptions:
        predicted_label = classify_description(description, glove_dict)
        if predicted_label is not None:
            y_true.append(true_label)
            y_pred.append(predicted_label)
            print(f"{filename:<40} {true_label:<15} {predicted_label:<15}")

    
    # Compute metrics
    if y_true and y_pred:
        precision = precision_score(y_true, y_pred, pos_label="ragebait")
        recall = recall_score(y_true, y_pred, pos_label="ragebait")
        f1 = f1_score(y_true, y_pred, pos_label="ragebait")
        accuracy = accuracy_score(y_true, y_pred)

        print(f"\nTotal number of entries classified: {len(y_true)}")
        print("\nMetrics:")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")
        print(f"Accuracy: {accuracy:.2f}")
    else:
        print("\nNo valid classifications made.")

# Example usage
if __name__ == "__main__":
    descriptions_folder = "descriptions"  # Path to folder containing description .txt files
    glove_path = "glove.840B.300d.txt"  # Path to original GloVe file
    retrofitted_path = "glove_retrofitted.txt"  # Path to save retrofitted GloVe file

    print("Loading GloVe vectors...")
    glove_dict = load_glove_vectors(glove_path)
    
    print("Evaluating model...")
    evaluate_model(descriptions_folder, glove_dict, retrofitted_path)
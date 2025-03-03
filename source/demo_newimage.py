import os
import base64
import numpy as np
from openai import OpenAI

api_key = ""

# Initialize OpenAI API client - Mine went here but I removed it for obvious reasons
client = OpenAI(api_key)

def encode_image(image_path):
    """Encode an image in base64 format."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def load_glove_vectors(glove_path):
    """Load retrofitted GloVe vectors."""
    glove_dict = {}
    with open(glove_path, "r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype="float32")
            glove_dict[word] = vector
    return glove_dict

def vectorize_text(text, glove_dict):
    """Vectorize text using GloVe vectors and return the centroid."""
    words = text.split()
    vectors = [glove_dict[word] for word in words if word in glove_dict]
    if vectors:
        return np.mean(vectors, axis=0)
    return None

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def classify_description(description, glove_dict):
    """Classify a description as ragebait or not ragebait using retrofitted GloVe vectors."""
    ragebait_vec = glove_dict.get("ragebait")
    non_ragebait_vec = glove_dict.get("non_ragebait")
    
    if ragebait_vec is None or non_ragebait_vec is None:
        raise ValueError("Key vectors for 'ragebait' or 'non_ragebait' not found in GloVe.")
    
    desc_vec = vectorize_text(description, glove_dict)
    if desc_vec is None:
        return None, "Unable to vectorize description."
    
    similarity_to_ragebait = cosine_similarity(desc_vec, ragebait_vec)
    similarity_to_non_ragebait = cosine_similarity(desc_vec, non_ragebait_vec)
    
    classification = "ragebait" if similarity_to_ragebait > similarity_to_non_ragebait else "not ragebait"
    return classification, f"Similarity to ragebait: {similarity_to_ragebait:.3f}, Similarity to not ragebait: {similarity_to_non_ragebait:.3f}"

def explain_classification(description, classification):
    """Ask ChatGPT to explain why the classification was made."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Here is the description: '{description}'. The classification is '{classification}'. Explain why this is the case."}
        ],
        max_tokens=300
    )
    return response.choices[0].message.content

def process_image(image_path, glove_path):
    """Process an image to classify it and explain the result."""
    # Encode the image
    base64_image = encode_image(image_path)
    
     # Generate description using GPT
    description_response = client.chat.completions.create(
        model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this meme image. Include any and all visible text, objects, and overall context/overtones. One line. No adjectives"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
    description = description_response.choices[0].message.content
    print(f"Description: {description}")
    
    # Load retrofitted GloVe vectors
    glove_dict = load_glove_vectors(glove_path)
    
    # Classify the description
    classification, similarity_details = classify_description(description, glove_dict)
    print(f"Classification: {classification}")
    #print(similarity_details)
    
    # Ask ChatGPT to explain the classification
    explanation = explain_classification(description, classification)
    print(f"Explanation: {explanation}")

# Example usage
if __name__ == "__main__":
    image_path = "image.jpg"  # Replace with new image path
    glove_path = "glove_retrofitted.txt"  
    
    process_image(image_path, glove_path)
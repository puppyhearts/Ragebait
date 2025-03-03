import base64
from openai import OpenAI
import os

api_key =""

# Set your OpenAI API key - Mine went here but I removed it for obvious reasons
client = OpenAI(api_key)

# Function to encode an image in base64 format
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to process images and generate descriptions
def process_images_in_folder(folder_path, max_images):
    # Ensure the descriptions folder exists
    os.makedirs("MAMI_summary", exist_ok=True)
    
    # Get list of images in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    # Limit the number of images processed
    image_files = image_files[:max_images]
    
    for image_name in image_files:
        # Define the path to the output description file
        output_file = os.path.join('MAMI_summary', f"{os.path.splitext(image_name)[0]}_description.txt")
        
        # Skip processing if description file already exists
        if os.path.exists(output_file):
            print(f"Description already exists for {image_name}. Skipping...")
            continue
        
        # Process the image
        image_path = os.path.join(folder_path, image_name)
        base64_image = encode_image(image_path)
        
        # Generate description using GPT
        response = client.chat.completions.create(
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
        
        # Extract and print the description
        description = response.choices[0].message.content
        print(f"Image: {image_name}")
        print(f"Description: {description}\n")
        
        # Save the description to a text file
        with open(output_file, "w") as f:
            f.write(description)

# Main execution
if __name__ == "__main__":
    # Specify folder path and max images
    folder_path = "MAMI_test"  # Replace with the path to your folder
    max_images = 499  # Adjust the number of images to process
    
    process_images_in_folder(folder_path, max_images)
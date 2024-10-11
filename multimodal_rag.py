from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from PIL import Image
from io import BytesIO
from datasets import load_dataset
import base64
import os

ds = load_dataset("GHonem/fashion_image_caption-3500")

# Create a directory to store the images
output_dir = 'images_dataset'
os.makedirs(output_dir, exist_ok=True)

# Initialize a list to store the new URIs (file paths)
image_paths = []

# Loop through the dataset and save each image to disk
cnt = 0 # for sample test
for idx, example in enumerate(ds['train']):
    if cnt == 50: # for sample test
        break
    image = example['image']  # Assuming this is a PIL.Image object or similar
    
    # Construct a file path
    image_path = os.path.join(output_dir, f'image_{idx}.png')  # You can change the extension if needed
    
    # Save the image to the file path
    if isinstance(image, Image.Image):
        image.save(image_path)
    
    # Append the file path (URI) to the list
    image_paths.append(image_path)
    cnt += 1

# Create a vector store and insert the image embeddings
vector_store = Chroma(
    collection_name="mm_rag_clip_photos", embedding_function=OpenCLIPEmbeddings()
)

vector_store.add_images(uris=image_paths)

# Querying with a text prompt using CLIP model
prompt = "green"
prompt_embedding = OpenCLIPEmbeddings().embed_query(prompt)

# Retrieve the most similar image
similar_images = vector_store.similarity_search_by_vector(prompt_embedding)

# Decode the base64 string
image_data = base64.b64decode(similar_images[0].page_content)

# Convert the binary data to an image
image = Image.open(BytesIO(image_data))

# Display or save the image
image.show()  # To display
image.save("decoded_image.png")  # To save the image to a file

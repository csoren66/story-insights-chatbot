import faiss
import numpy as np

def create_faiss_index(chunks, model_name='distilbert-base-uncased'):
    from sentence_transformers import SentenceTransformer
    
    # Load the model
    model = SentenceTransformer(model_name)
    
    # Create embeddings for each chunk
    embeddings = np.array([model.encode(chunk) for chunk in chunks]).astype('float32')
    
    # Initialize FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # Euclidean distance-based index
    index.add(embeddings)
    
    return index

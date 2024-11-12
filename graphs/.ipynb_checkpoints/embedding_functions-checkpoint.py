

from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Set the model to evaluation mode to prevent updates to weights
model.eval()

def embedding_func(speech_line):
    """
    Generate BERT embedding for a given speech line.

    Parameters:
    - speech_line: A string representing a single line of speech.

    Returns:
    - A 768-dimensional BERT embedding as a list of floats.
    """
    # Tokenize the input text and convert to PyTorch tensors
    inputs = tokenizer(speech_line, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # Get the model outputs (embedding) from the last hidden layer
    with torch.no_grad():
        outputs = model(**inputs)
    
    # The embeddings are in 'last_hidden_state', take the [CLS] token as the sentence-level embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
    
    # Convert to a Python list (detaching from the computation graph)
    embedding = cls_embedding.squeeze().tolist()

    return embedding




# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# def embedding_func_2 (speech_line):
#     """
#     Generate new embeddings for a line fo speech
#     """
#     # Tokenize the input text and convert to PyTorch tensors
#     wrapped_speech = [speech_line]
#     embedding = model.encode(wrapped_speech)
   
#     return embedding[0]
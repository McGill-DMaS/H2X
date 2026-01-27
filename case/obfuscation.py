import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, cosine_similarity

strings_dict = {
    "S1": "This is a normal english sentence.",
"S2": "Second sentence is related to software and information science.",
"S3": "This is a malicious file related malware and other attacks.",

"S1 (French)": "C'est une phrase anglaise normale.",
"S1 (Russian)": "Это обычное английское предложение.",
"S1 (Polish)": "To jest normalne angielskie zdanie." ,

"S2 (French)": "La deuxième phrase est liée aux logiciels et aux sciences de l information.",
"S2 (Russian)":  "Второе предложение связано с программным обеспечением и информатикой.",
"S2 (Polish)": "Drugie zdanie dotyczy oprogramowania i informatyki.",

"S3 (French)": "Il s'agit d'un logiciel malveillant lié à un fichier et d'autres attaques.",
"S3 (Russian)": "Это вредоносное ПО, связанное с файлом, и другие атаки.",
"S3 (Polish)":"Jest to złośliwe oprogramowanie związane ze złośliwym plikiem i inne ataki."

}

# Load the SBERT model
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Define the list of keys to encode
keys = list(strings_dict.keys())

# Encode the strings associated with each key using SBERT
embeddings = model.encode(list(strings_dict.values()))

# Calculate the cosine distance matrix between the encoded vectors
cosine_matrix = cosine_similarity(embeddings)

# Create a Pandas DataFrame from the cosine distance matrix
df = pd.DataFrame(cosine_matrix, columns=keys, index=keys)

# Export the DataFrame to a CSV file
df.to_csv('cosine_distance_matrix.csv')
print("completed...!!!")

"""
This case study is not successful should be consider later
"""

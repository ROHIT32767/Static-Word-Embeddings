import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def load_embeddings(embedding_path):
    """Load embeddings from a file into a dictionary."""
    embeddings = {}
    with open(embedding_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[word] = vector
    return embeddings

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def main(embedding_path):
    embeddings = load_embeddings(embedding_path)
    wordsim_df = pd.read_csv('WordSim-353.csv')
    dataset_scores = []
    cosine_similarities = []
    results = []
    
    for _, row in wordsim_df.iterrows():
        word1 = row['Word 1']
        word2 = row['Word 2']
        score = row['Human (Mean)']
        
        if word1 not in embeddings or word2 not in embeddings:
            continue
        
        vec1 = embeddings[word1]
        vec2 = embeddings[word2]
        sim = cosine_similarity(vec1, vec2)
        
        dataset_scores.append(score)
        cosine_similarities.append(sim)
        results.append([word1, word2, sim])
    
    correlation, _ = spearmanr(dataset_scores, cosine_similarities)
    print(f"Spearman's Rank Correlation: {correlation:.4f}")
    
    # Create CSV filename based on embedding file
    csv_filename = os.path.splitext(os.path.basename(embedding_path))[0] + ".csv"
    
    # Save results to CSV
    results_df = pd.DataFrame(results, columns=['Word 1', 'Word 2', 'Cosine Similarity'])
    results_df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: wordsim.py <embedding_path>.txt")
        sys.exit(1)

    embedding_path = sys.argv[1]
    main(embedding_path)

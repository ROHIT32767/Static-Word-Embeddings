# INLP Assignment 3

## Generating Word embeddings

Data used: [link](https://iiitaphyd-my.sharepoint.com/personal/advaith_malladi_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fadvaith%5Fmalladi%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FiNLP%5FA2%2FiNLP%2DA2%2Ezip&parent=%2Fpersonal%2Fadvaith%5Fmalladi%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FiNLP%5FA2&ga=1).


### SVD Embeddings

1. **svd_utils.py:** contains the code to generate SVD embeddings.
2. **svd.py:** contains the code to load data and calls the functions in `svd_utils.py` to generate embeddings.

    To generate SVD embeddings run the following command:
    ```
    python3 svd.py
    ```

    The embeddings will be saved in the current directory.

### Word2Vec (SGNS) Embeddings

1. **skip_gram_utils.py:** contains the functions to preprocess and generate skip-gram embeddings.
2. **skip-gram.py:** contains the code to load data and calls the functions in `skip_gram_utils.py` to generate embeddings.

    To generate SGNS embeddings run the following command:
    ```
    python3 skip-gram.py
    ```

    The embeddings will be saved in the current directory.


## Classification Model

Saved embeddings can be downloaded from this [link.](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/kukkapalli_shravya_students_iiit_ac_in/EsLrLo5RAY9DiRvJ_LKicE0BlZDRkLN2Y_K0mcBnk9x-Hw?e=UvF8N4)

**lstm_utils.py:** contains the code to load data and train the LSTM model.

**svd-classification.py:** contains the code to call the functions in `lstm_utils.py` to train the LSTM model using SVD embeddings.

**skip-gram-classification.py:** contains the code to call the functions in `lstm_utils.py` to train the LSTM model using SGNS embeddings.

### Using SVD Embeddings
1. Download the pretrained embeddings `word_embeddings_svd_best.pt`, `word2idx_svd.pt` and `idx2word_svd.pt` from the provided link.
2. Run the following command to train the model using SVD embeddings:
    ```
    python3 svd-classification.py
    ```

### Using Word2Vec Embeddings
1. Download the pretrained embeddings `word_embeddings_skip_gram_best.pt`, `word2idx_skip_gram.pt` and `idx2word_skip_gram.pt` from the provided link.
2. Run the following command to train the model using SVD embeddings:
    ```
    python3 skip-gram-classification.py
    ```
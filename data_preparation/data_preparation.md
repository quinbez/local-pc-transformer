# Data Preparation Pipeline for PC-Transformer

This document explains how raw Penn Treebank (PTB) text is transformed into model-ready tensors for training, validation, and testing.

## 1. Raw Data

We start with the Penn Treebank dataset, stored as plain text files:

- `train.txt`
- `valid.txt`
- `test.txt`

These are located in: `data_preparation/data/ptb/`.

## 2. Tokenizer Training

**File:** `prepare_tokens.py`

- We train a Byte Pair Encoding (BPE) tokenizer on the training corpus.  
- Special tokens (`[UNK]`, `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`) are reserved.  
- The tokenizer vocabulary size is set in `config.py` (`vocab_size = 4000`).  
- After training, the tokenizer is saved to `data_preparation/tokenizer.json`. 

## 3. Encoding Text into Token IDs

Still in **`prepare_tokens.py`**:

- Each dataset split (`train`, `valid`, `test`) is read as text.  
- The tokenizer encodes text into integer token IDs.  
- The IDs are converted into PyTorch tensors.  
- Saved to:

    ``` bash 
    data_preparation/encoded/train.pt
    data_preparation/encoded/valid.pt
    data_preparation/encoded/test.pt
    ```

These `.pt` files contain **1D tensors of token IDs**.

## 4. Sequence Construction

**File:** `dataset.py`

- `EncodedDataset` loads the saved `.pt` token tensors.  
- Tokens are divided into **fixed-length sequences** of `max_len + 1`.  
- Each sequence is split into:
  - `input_ids` → first `max_len` tokens  
  - `target_ids` → shifted by one token (next-token prediction)  

This prepares data for **autoregressive training** (predict the next token given the previous tokens).

## 5. Data Loaders

**File:** `data_utils.py`

- `get_datasets()` loads train/valid/test splits as `EncodedDataset` objects.  
- `get_loaders()` wraps them into PyTorch `DataLoader`s:
  - Training loader shuffles batches.  
  - Validation/test loaders preserve order.  

# ai4india: LLM Training, Inference, and Fine-Tuning

## Overview

This project provides a comprehensive framework for training, inferencing, and fine-tuning Language Models (LLMs) on custom datasets. It leverages PyTorch and the Hugging Face Transformers library to implement a decoder-only transformer model.

## Features

*   **Decoder-Only Transformer Model:** Implementation of a decoder-only transformer architecture suitable for language modeling tasks.
*   **Customizable Hyperparameters:** Easily configurable model and training parameters, including embedding dimension, number of attention heads, and number of transformer blocks.
*   **Efficient Data Handling:** Utilizes iterable datasets and data loaders for memory-efficient processing of large text datasets.
*   **Training and Validation:** Includes training and validation loops with loss and perplexity monitoring.
*   **Text Generation:** Provides a text generation function for sampling from the trained model.
*   **Model Saving and Loading:** Functionality to save and load trained models and tokenizers.

## Features

- ✅ Decoder-Only Transformer Model
- ✅ Customizable Hyperparameters
- ✅ Efficient Data Handling
- ✅ Training and Validation
- ✅ Text Generation
- ✅ Model Saving and Loading
- ✅ Streaming data from storage to main memory
- ✅ Training script
- ✅ Inference script
- ✅ Data downloading

## Not Implemented Features

- [ ] KV Cache Optimization
- [ ] Dynamic Batching
- [ ] Prefill
- [ ] Speculative Decoding

## Requirements

*   Python 3.7+
*   PyTorch
*   Hugging Face Transformers
*   tqdm
*   regex
*   urllib

You can install the necessary dependencies using `pip`:

```bash
pip install -r requirements.txt
```

A `requirements.txt` file is provided in this repository with all the required dependencies.

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd ai4india
    ```

2.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Preparation

*   The project expects a text dataset in a `.tar.gz` archive containing `train.txt` and `test.txt` files. Each line in these files should represent a sentence.
*   You can specify the data URL in the `train.py` script.

### 2. Training the Model

To train the model, run the `train.py` script:

```bash
python train.py
```

*   **Hyperparameters:** The training script uses hyperparameters defined in the `get_hyperparameters` function within `utils/model_utils.py`. You can modify these values to customize the training process. Key hyperparameters include:
    *   `emb_dim`: Embedding dimension
    *   `num_heads`: Number of attention heads
    *   `num_blocks`: Number of transformer blocks
    *   `batch_size`: Batch size
    *   `learning_rate`: Learning rate
    *   `num_epochs`: Number of epochs
    *   `context_size`: Maximum sequence length

*   **Training Process:** The script downloads the dataset (if not already present), preprocesses it, and trains the decoder language model. It also performs periodic validation and saves the trained model and tokenizer to the `models` directory.

### 3. Testing the Model

To test the trained model, run the `test.py` script:

```bash
python test.py
```

*   The script loads the trained model and tokenizer from the `models` directory and generates text based on predefined prompts.

### 4. Model Details

#### Files

*   `transformer.py`: This file is currently empty. It could be used to define custom transformer components or configurations.
*   `train.py`: Contains the main training loop, data loading, model initialization, and saving logic.
*   `test.py`: Contains the model testing and text generation logic.
*   `utils/data_utils.py`: Implements data loading, preprocessing, and batching utilities.
*   `utils/model_utils.py`: Implements model definition, weight initialization, training utilities (loss computation, perplexity), and saving/loading functions.

#### Model Architecture

The core of this project is the `DecoderLanguageModel` class, defined in `utils/model_utils.py`. It consists of the following components:

*   **Embedding Layer:** Maps input tokens to high-dimensional embeddings.
*   **Decoder Blocks:** Stacked transformer decoder blocks, each containing:
    *   **RMSNorm:** Root Mean Square Layer Normalization for stable training.
    *   **MultiHeadAttention:** Multi-head self-attention mechanism.
    *   **MLP:** A multi-layer perceptron (feed-forward network).
*   **Output Layer:** Projects the final hidden states to the vocabulary space.

#### Key Functions

*   **`download_and_prepare_data(url, batch_size, tokenizer, max_length)` (in `utils/data_utils.py`):** Downloads the dataset, extracts the training and testing files, and creates data loaders.
*   **`DecoderLanguageModel(vocab_size, emb_dim, num_heads, num_blocks, pad_idx)` (in `utils/model_utils.py`):** Initializes the decoder-only transformer model.
*   **`generate_text(model, start_string, tokenizer, device, max_length)` (in `utils/model_utils.py`):** Generates text from a given starting string using the trained model.
*   **`save_model(model, tokenizer, model_name)` (in `utils/model_utils.py`):** Saves the trained model and tokenizer.
*   **`load_model(model_name, device=None)` (in `utils/model_utils.py`):** Loads a pre-trained model and tokenizer.

## Customization

*   **Dataset:** To use your own dataset, modify the `download_and_prepare_data` function in `utils/data_utils.py` to load and preprocess your data.
*   **Model Architecture:** You can modify the `DecoderLanguageModel` class in `utils/model_utils.py` to experiment with different model architectures.
*   **Training Parameters:** Adjust the hyperparameters in the `get_hyperparameters` function in `utils/model_utils.py` to optimize training for your specific dataset and task.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


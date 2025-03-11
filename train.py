from utils.data_utils import *
from utils.model_utils import *


if __name__ == "__main__":
    # Initialize random seeds to ensure reproducible results
    set_seed(42)

    # Retrieve model architecture and training hyperparameters from configuration
    # emb_dim: dimensionality of input token and intermediary embeddings
    # num_heads: number of attention heads in each transformer block
    # num_blocks: number of transformer blocks in the model
    # batch_size: mini-batch size
    # learning_rate: step size for optimizer updates
    # num_epochs: number of complete passes through the training dataset
    # context_size: maximum input sequence length
    emb_dim, num_heads, num_blocks, batch_size, learning_rate, num_epochs, context_size = get_hyperparameters()

    # Initialize the tokenizer using Microsoft's Phi-3.5-mini model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    # Get padding token index for padding shorter sequences
    pad_idx = tokenizer.pad_token_id

    # Check for CUDA-capable GPU and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download the news dataset and create DataLoader objects for training and testing
    # DataLoaders handle batching and shuffling
    data_url = "https://www.thelmbook.com/data/news"
    train_dataloader, test_dataloader = download_and_prepare_data(
        data_url, batch_size, tokenizer, context_size
    )

    # Get the size of the vocabulary that the model needs to handle
    vocab_size = len(tokenizer)
    print(f"\nVocabulary size: {vocab_size}\n")

    # Initialize the Decoder language model with specified architecture parameters
    # vocab_size: determines output layer dimensionality
    # emb_dim: size of token embeddings and intermediary embeddings
    # num_heads: number of attention heads per transformer block
    # num_blocks: number of transformer blocks in the model
    # pad_idx: special token ID used for padding shorter sequences
    model = DecoderLanguageModel(
        vocab_size, emb_dim, num_heads, num_blocks, pad_idx
    )

    # Move the model to GPU if available
    model.to(device)

    # Initialize model weights using custom initialization scheme
    # This is important for stable training of deep neural networks
    initialize_weights(model)

    # Initialize the AdamW optimizer with specified learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Initialize the loss function (Cross Entropy) for training
    # ignore_index=pad_idx ensures that padding tokens don't contribute to the loss
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Calculate and display the total number of trainable parameters in the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params}\n")

    # Set evaluation interval (number of examples after which to perform validation)
    # 200,000 examples provides a good balance between training time and monitoring frequency
    eval_interval = 200_000
    examples_processed = 0  # Counter for tracking progress toward next evaluation

    # Define test contexts for generating sample text during evaluation
    contexts = [
        "Moscow",
        "New York",
        "A hurricane",
        "The President"
    ]

    # Main training loop - iterate through specified number of epochs
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()

        # Initialize tracking variables for this epoch
        total_loss = 0.0      # Accumulator for loss across all batches
        total_tokens = 0      # Counter for actual tokens processed (excluding padding)

        # Create progress bar for monitoring training progress
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        # Iterate through batches in the training data
        for batch_idx, (input_seq, target_seq) in enumerate(progress_bar):
            # Move input and target sequences to GPU if available
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            # Clear gradients from previous batch
            optimizer.zero_grad()

            # Forward pass: get model predictions for this batch
            # output shape: (batch_size, seq_len, vocab_size)
            logits = model(input_seq)

            # Reshape logits and target tensors for loss computation
            logits = logits.reshape(-1, logits.size(-1))
            target = target_seq.reshape(-1)

            # Create mask to exclude padding tokens from loss calculation
            mask = target != pad_idx

            # Compute loss between model predictions and actual targets
            # Using masked versions to ignore padding tokens
            loss = criterion(logits[mask], target[mask])

            # Backward pass: compute gradients of loss with respect to model parameters
            loss.backward()

            # Update model parameters using calculated gradients
            optimizer.step()

            # Calculate actual loss value for this batch accounting for padding
            loss_value = loss.item() * mask.sum().item()

            # Accumulate total loss and tokens for epoch statistics
            total_loss += loss_value
            total_tokens += mask.sum().item()
            examples_processed += input_seq.size(0)

            # Update progress bar with current batch loss
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Periodic evaluation after processing specified number of examples
            if examples_processed >= eval_interval:
                # Calculate average loss over the last eval_interval examples
                avg_loss = total_loss / total_tokens
                print(f"\nAfter {examples_processed} examples, Average Loss: {avg_loss:.4f}")

                # Switch to evaluation mode
                model.eval()

                # Compute validation metrics
                average_loss, perplexity = compute_loss_and_perplexity(
                    model, test_dataloader, tokenizer, criterion, device, max_sentences=1000
                )
                # Record validation
                print(f"\nValidation Average Loss: {average_loss:.4f}, Perplexity: {perplexity:.2f}")

                model.eval()

                # Generate sample texts to qualitatively assess model performance
                for context in contexts:
                    # Generate text continuation for each test context
                    generated_text = generate_text(
                        model=model,
                        start_string=context,
                        tokenizer=tokenizer,
                        device=device,
                        max_length=50
                    )
                    print(f"\nContext: {context}")
                    print(f"\nGenerated text: {generated_text}\n")

                # Switch back to training mode for continued training
                model.train()

                # Reset counters for next evaluation interval
                examples_processed = 0
                total_loss = 0.0
                total_tokens = 0

        # End-of-epoch reporting
        if total_tokens > 0:
            # Calculate and display average loss for the epoch
            avg_loss = total_loss / total_tokens
            print(f"\nEpoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        else:
            # Handle edge case where no tokens were processed
            print(f"\nEpoch {epoch+1}/{num_epochs} completed.")

        # Perform end-of-epoch validation
        model.eval()

        # Generate sample texts for qualitative assessment
        print("\nGenerating text based on contexts using generate_text:\n")
        for context in contexts:
            generated_text = generate_text(
                model=model,
                start_string=context,
                tokenizer=tokenizer,
                device=device,
                max_length=50
            )
            print(f"\nContext: {context}")
            print(f"\nGenerated text: {generated_text}\n")

        average_loss, perplexity = compute_loss_and_perplexity(
            model, test_dataloader, tokenizer, criterion, device, max_sentences=1000
        )
        print(f"\nValidation Average Loss: {average_loss:.4f}, Perplexity: {perplexity:.2f}")

        # Reset to training mode for next epoch
        model.train()

    # Save the trained model and tokenizer for later use
    # This includes model architecture, weights, and tokenizer configuration
    model_name = "Decoder_LM"
    save_model(model, tokenizer, model_name)
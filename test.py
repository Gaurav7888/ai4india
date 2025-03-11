from utils.data_utils import *
from utils.model_utils import *

if __name__ == "__main__":

    model_name = "Decoder_LM"

    # Load the previously saved model and tokenizer from disk
    # This recreates the exact model state from after training
    model, tokenizer = load_model(model_name)

    model.eval()

    # Print header for test section
    print("\nTesting the model:\n")

    # Define a list of test prompts to evaluate model performance
    contexts = [
        "Moscow",
        "New York",
        "A hurricane",
        "The President"
    ]

    # Iterate through each test prompt and generate text
    for context in contexts:
        # Generate text using greedy decoding (most likely tokens)
        generated_text = generate_text(
            model=model,          # The loaded language model
            start_string=context, # Text to continue
            tokenizer=tokenizer,  # Tokenizer for text conversion
            device=device,        # CPU or GPU device
            max_length=50         # Maximum length of generated sequence
        )
        # Print the original prompt and model's response
        print(f"\nPrompt: {context}")
        print(f"\nGenerated response: {generated_text}\n")
import torch

def create_mask_attention(input_ids, mask_token_id):
    """
    Creates attention mask where masked tokens can attend to all tokens,
    but other tokens cannot attend to masked tokens.
    
    input_ids: tensor of shape [batch_size, sequence_length]
    mask_token_id: integer ID of the mask token
    returns: attention mask of shape [batch_size, 1, sequence_length, sequence_length]
    """
    batch_size, seq_length = input_ids.shape
    
    # Create mask where 1 = not mask token, 0 = mask token
    mask = (input_ids != mask_token_id).float()  # [batch_size, seq_length]
    
    # Create attention mask by outer product
    attention_mask = mask.unsqueeze(-1) @ mask.unsqueeze(-2)  # [batch_size, seq_length, seq_length]
    
    # Add batch dimension for attention heads
    attention_mask = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_length, seq_length]
    
    return attention_mask

def visualize_attention_mask(input_ids, mask_token_id):
    """
    Visualize the attention mask for given input IDs.
    
    input_ids: tensor of shape [batch_size, sequence_length]
    mask_token_id: integer ID of the mask token
    """
    attention_mask = create_mask_attention(input_ids, mask_token_id)
    mask = attention_mask.squeeze(0).squeeze(0)
    
    print(f"\nInput IDs:\n{input_ids}")
    print(f"\nMask token ID: {mask_token_id}")
    print("\nAttention Mask:")
    
    seq_length = input_ids.size(-1)
    
    # Print column headers
    print("\n{:>5}".format(""), end=" ")
    for j in range(seq_length):
        print(f"{j:>5}", end=" ")
    print("\n" + "-" * (5 + 6 * seq_length))
    
    # Print rows
    for i in range(seq_length):
        print(f"{i:>5}", end=" ")
        for j in range(seq_length):
            print(f"{mask[i,j]:>5.0f}", end=" ")
        print()

if __name__ == "__main__":
    # Example usage with dummy data
    batch_size, seq_length = 1, 10
    vocab_size = 65  # Character vocab size
    mask_token_id = vocab_size - 1  # Last token is mask
    
    # Create random sequence
    input_ids = torch.randint(0, vocab_size-1, (batch_size, seq_length))
    
    # Set some positions to mask token
    mask_positions = [2, 5, 8]
    input_ids[0, mask_positions] = mask_token_id
    
    visualize_attention_mask(input_ids, mask_token_id)
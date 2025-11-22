import torch
import torch.nn as nn
import numpy as np
import yaml
from typing import Dict, Any, List
import sys
import os

# Add the necessary paths to import modules
sys.path.append('.')

# Import the required modules
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from models.losses import ACTLossHead
from utils.functions import load_model_class

def preprocess_sudoku_input(sudoku_string: str) -> np.ndarray:
    """
    Convert a Sudoku string to the model's input format.
    
    Args:
        sudoku_string: Multi-line string with underscores for empty cells
        Example:
        '''
        _ _ _ 9 _ _ 8 3 1
        _ 9 6 8 _ 7 _ _ _
        _ _ _ _ 3 _ 5 _ _
        _ _ 6 8 _ _ _ _ _
        7 4 _ _ _ 6 _ 2 3
        _ _ _ _ _ 9 _ 4 _
        2 _ _ _ 4 _ _ 1 _
        6 _ _ _ 2 _ 1 5 7
        '''
    
    Returns:
        Flattened array of shape (81,) with values 1-10 (0=pad, 1=blank, 2=1, 3=2, ..., 10=9)
    """
    # Parse the multi-line string
    lines = [line.strip() for line in sudoku_string.strip().split('\n') if line.strip()]
    
    # Convert to 9x9 grid
    grid = []
    for line in lines:
        row = []
        for char in line.split():
            if char == '_' or char == '.' or char == '0':
                row.append(1)  # Blank cell -> 1 (since 0 is pad_id)
            else:
                # Convert digit to model's vocabulary: digit -> digit + 1
                row.append(int(char) + 1)
        grid.append(row)
    
    # Flatten to 1D array (81 elements)
    flattened = np.array(grid).flatten()
    return flattened

def postprocess_sudoku_output(predictions: np.ndarray) -> np.ndarray:
    """
    Convert model predictions back to Sudoku digits (1-9).
    
    Args:
        predictions: Model output of shape (81,) with values 1-10
        
    Returns:
        Sudoku grid of shape (9, 9) with values 1-9
    """
    # Convert from model vocabulary to digits: value -> value - 1
    # But clip to 1-9 range since 0 would be blank/pad
    digits = np.clip(predictions - 1, 1, 9)
    return digits.reshape(9, 9)

def create_model_from_config(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """
    Create and load the trained model.
    
    Args:
        config_path: Path to all_config.yaml
        checkpoint_path: Path to model checkpoint (e.g., step_10416)
        device: Device to run inference on
    
    Returns:
        Loaded model ready for inference
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Extract model configuration
    arch_config = config_data['arch']
    model_config = {
        **arch_config,
        'batch_size': 1,  # Single example inference
        'vocab_size': 11,  # From dataset.json
        'seq_len': 81,     # From dataset.json
        'num_puzzle_identifiers': 1,  # From dataset.json
        'causal': False
    }
    
    # Remove loss from model config as it's handled separately
    model_config.pop('loss', None)
    
    # Create model
    model_cls = load_model_class(arch_config['name'])
    loss_head_cls = load_model_class(arch_config['loss']['name'])
    
    # Create base model
    base_model = model_cls(model_config)
    
    # Wrap with loss head (this is how it was trained)
    model = loss_head_cls(base_model, **arch_config['loss'].get('__pydantic_extra__', {}))
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle compiled model prefix if present
    state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('_orig_mod.'):
            state_dict[k[10:]] = v  # Remove '_orig_mod.' prefix
        else:
            state_dict[k] = v
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model

def solve_sudoku(model: nn.Module, sudoku_input: str, device: str = "cuda") -> np.ndarray:
    """
    Solve a Sudoku puzzle using the trained model.
    
    Args:
        model: Loaded TRM model
        sudoku_input: Sudoku puzzle as multi-line string
        device: Device to run inference on
    
    Returns:
        Solved Sudoku as 9x9 numpy array
    """
    # Preprocess input
    input_array = preprocess_sudoku_input(sudoku_input)
    
    # Create batch in the expected format
    batch = {
        'inputs': torch.from_numpy(input_array).unsqueeze(0).to(device),  # Shape: (1, 81)
        'labels': torch.zeros((1, 81), dtype=torch.long).to(device),      # Dummy labels
        'puzzle_identifiers': torch.zeros((1,), dtype=torch.long).to(device)  # Single puzzle ID
    }
    
    # Run inference
    with torch.no_grad():
        # Initialize model state
        carry = model.initial_carry(batch)
        
        # Run until model halts (similar to evaluation loop)
        inference_steps = 0
        max_steps = 50  # Safety limit
        
        while inference_steps < max_steps:
            carry, loss, metrics, preds, all_finish = model(
                carry=carry, 
                batch=batch, 
                return_keys=['preds']  # We only need predictions
            )
            inference_steps += 1
            
            if all_finish:
                break
        
        print(f"Solved in {inference_steps} steps")
        
        # Extract predictions
        predictions = preds['preds'].cpu().numpy()[0]  # Shape: (81,)
    
    # Postprocess to get final Sudoku solution
    solution = postprocess_sudoku_output(predictions)
    return solution

def print_sudoku_grid(grid: np.ndarray, title: str = "Sudoku"):
    """
    Pretty print a Sudoku grid.
    
    Args:
        grid: 9x9 numpy array
        title: Title for the grid
    """
    print(f"\n{title}:")
    print("-" * 25)
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 25)
        row = " | ".join(
            " ".join(str(grid[i, j]) for j in range(3 * k, 3 * (k + 1)))
            for k in range(3)
        )
        print(f"| {row} |")
    print("-" * 25)

def main():
    # Configuration
    config_path = "checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_att_sudoku_l40s_first/all_config.yaml"
    checkpoint_path = "checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_att_sudoku_l40s_first/step_10416"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Your example Sudoku input
    sudoku_input = """
    _ _ _ 9 _ _ 8 3 1
    _ 9 6 8 _ 7 _ _ _
    _ _ _ _ 3 _ 5 _ _
    _ _ 6 8 _ _ _ _ _
    7 4 _ _ _ 6 _ 2 3
    _ _ _ _ _ 9 _ 4 _
    2 _ _ _ 4 _ _ 1 _
    6 _ _ _ 2 _ 1 5 7
    """
    
    # Expected solution for comparison
    expected_solution = np.array([
        [5, 2, 6, 7, 9, 4, 8, 3, 1],
        [3, 9, 1, 2, 6, 8, 4, 7, 5],
        [4, 8, 7, 3, 1, 5, 2, 9, 6],
        [1, 6, 8, 5, 3, 2, 7, 4, 9],
        [9, 3, 5, 4, 7, 6, 1, 8, 2],
        [7, 4, 2, 9, 8, 1, 5, 6, 3],
        [8, 7, 3, 1, 5, 9, 6, 2, 4],
        [2, 5, 9, 6, 4, 7, 3, 1, 8],
        [6, 1, 4, 8, 5, 3, 9, 5, 7]
    ])
    
    try:
        # Load model
        print("Loading model...")
        model = create_model_from_config(config_path, checkpoint_path, device)
        
        # Solve Sudoku
        print("Solving Sudoku...")
        solution = solve_sudoku(model, sudoku_input, device)
        
        # Display results
        print_sudoku_grid(solution, "Model Solution")
        print_sudoku_grid(expected_solution, "Expected Solution")
        
        # Calculate accuracy
        input_array = preprocess_sudoku_input(sudoku_input)
        input_grid = postprocess_sudoku_output(input_array)
        
        # Count how many predicted cells match expected (only check originally blank cells)
        blank_positions = input_grid == 0
        correct_predictions = np.sum(solution[blank_positions] == expected_solution[blank_positions])
        total_blanks = np.sum(blank_positions)
        
        accuracy = correct_predictions / total_blanks if total_blanks > 0 else 1.0
        print(f"\nAccuracy on blank cells: {accuracy:.2%} ({correct_predictions}/{total_blanks})")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
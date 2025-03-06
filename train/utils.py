import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_memory(model, batch_size, input_shape, dtype=torch.float32):
    bytes_per_param = torch.finfo(dtype).bits // 8  # Bytes per float
    total_params = sum(p.numel() for p in model.parameters())

    # Parameter memory (weights + gradients)
    param_memory = total_params * bytes_per_param * 2  # 2x for gradients

    # Activation memory (assumes full batch stored)
    dummy_input = torch.randn(batch_size, *input_shape, dtype=dtype)
    with torch.no_grad():
        output = model(dummy_input)
    activation_memory = output.numel() * bytes_per_param  # Approximate

    # Optimizer state (Adam = 2x model size)
    optimizer_memory = total_params * bytes_per_param * 2

    total_memory = param_memory + activation_memory + optimizer_memory
    return total_memory / (1024**2)  # Convert to MB
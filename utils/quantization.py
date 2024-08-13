import torch
import torch.nn as nn
import torch.quantization
from utils.logger import logger
from src.train import test
import copy
import time
import os

def quantize_model(model, device, test_loader):
    """
    Quantize the given model and compare its performance with the original model.

    Args:
    - model (nn.Module): The original model to be quantized
    - device (torch.device): The device to run the models on
    - test_loader (DataLoader): DataLoader for the test dataset

    Returns:
    - dict: A dictionary containing the original and quantized models, along with their performances
    """
    # Create a copy of the original model for quantization
    quantized_model = copy.deepcopy(model)

    # Set the model to eval mode
    model.eval()
    quantized_model.eval()

    # Fuse Conv, BN and ReLU layers
    quantized_model = torch.quantization.fuse_modules(quantized_model, [['conv1', 'relu'], ['conv2', 'relu']])

    # Specify quantization configuration
    quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(quantized_model, inplace=True)

    # Calibrate the model with the test data
    with torch.no_grad():
        for data, _ in test_loader:
            quantized_model(data.to(device))

    # Convert to quantized model
    torch.quantization.convert(quantized_model, inplace=True)

    # Move models to the appropriate device
    model = model.to(device)
    quantized_model = quantized_model.to(device)

    # Evaluate both models
    original_loss, original_accuracy = test(model, device, test_loader)
    quantized_loss, quantized_accuracy = test(quantized_model, device, test_loader)

    # Measure inference time
    def measure_inference_time(model, device, test_loader):
        start_time = time.time()
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                _ = model(data)
        end_time = time.time()
        return end_time - start_time

    original_inference_time = measure_inference_time(model, device, test_loader)
    quantized_inference_time = measure_inference_time(quantized_model, device, test_loader)

    # Calculate model sizes
    def get_model_size(model):
        torch.save(model.state_dict(), "temp.p")
        size = os.path.getsize("temp.p") / 1e6  # Size in MB
        os.remove('temp.p')
        return size

    original_size = get_model_size(model)
    quantized_size = get_model_size(quantized_model)

    results = {
        'original_model': model,
        'quantized_model': quantized_model,
        'original_accuracy': original_accuracy,
        'quantized_accuracy': quantized_accuracy,
        'original_loss': original_loss,
        'quantized_loss': quantized_loss,
        'original_inference_time': original_inference_time,
        'quantized_inference_time': quantized_inference_time,
        'original_size': original_size,
        'quantized_size': quantized_size
    }

    logger.info(f"Original Model - Accuracy: {original_accuracy:.2f}%, Loss: {original_loss:.4f}, "
                f"Inference Time: {original_inference_time:.4f}s, Size: {original_size:.2f}MB")
    logger.info(f"Quantized Model - Accuracy: {quantized_accuracy:.2f}%, Loss: {quantized_loss:.4f}, "
                f"Inference Time: {quantized_inference_time:.4f}s, Size: {quantized_size:.2f}MB")

    return results

def analyze_quantization_results(results):
    """
    Analyze and print the results of quantization.

    Args:
    - results (dict): The results dictionary from quantize_model function

    Returns:
    - str: A formatted string containing the analysis
    """
    accuracy_change = results['quantized_accuracy'] - results['original_accuracy']
    loss_change = results['quantized_loss'] - results['original_loss']
    inference_speedup = results['original_inference_time'] / results['quantized_inference_time']
    size_reduction = (results['original_size'] - results['quantized_size']) / results['original_size'] * 100

    analysis = f"""
    Quantization Analysis:
    ----------------------
    Accuracy Change: {accuracy_change:.2f}%
    Loss Change: {loss_change:.4f}
    Inference Speedup: {inference_speedup:.2f}x
    Model Size Reduction: {size_reduction:.2f}%

    The quantized model {'improved' if accuracy_change > 0 else 'decreased'} accuracy by {abs(accuracy_change):.2f}%.
    The loss {'increased' if loss_change > 0 else 'decreased'} by {abs(loss_change):.4f}.
    Inference is {inference_speedup:.2f}x faster with the quantized model.
    The quantized model is {size_reduction:.2f}% smaller than the original model.
    """

    logger.info(analysis)
    return analysis
# File: main.py

import torch
import torch.optim as optim
import os
from datetime import datetime
import argparse

from src.model import CNN, create_lora_cnn, StandardLoRALinear, ScaledLoRALinear, MultiRankLoRALinear
from src.data import load_mnist, load_fashion_mnist
from src.train import train_and_evaluate, fine_tune
from utils.visualization import (plot_learning_curves, plot_weight_heatmaps, 
                                 plot_lora_weights, visualize_feature_space, 
                                 plot_confusion_matrix, plot_grad_cam)
from utils.analysis import hyperparameter_tuning, compare_models
from utils.logger import logger

def parse_arguments():
    parser = argparse.ArgumentParser(description="miniLora: Low-Rank Adaptation experiment")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Initial learning rate")
    parser.add_argument('--mnist_epochs', type=int, default=1, help="Number of epochs for MNIST training")
    parser.add_argument('--fashion_mnist_epochs', type=int, default=1, help="Number of epochs for Fashion MNIST fine-tuning")
    parser.add_argument('--output_dir', type=str, default=None, help="Output directory for results (default: timestamped folder)")
    parser.add_argument('--no_cuda', action='store_true', help="Disable CUDA even if available")
    parser.add_argument('--skip_hyperparameter_tuning', action='store_true', help="Skip hyperparameter tuning")
    return parser.parse_args()

def main(args):
    # Set up device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_cuda else "cpu")

    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_DIR = f"output_{timestamp}"
    else:
        OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Load datasets
    mnist_train_loader, mnist_test_loader = load_mnist(batch_size=args.batch_size)
    fashion_train_loader, fashion_test_loader = load_fashion_mnist(batch_size=args.batch_size)

    # Hyperparameter tuning
    if not args.skip_hyperparameter_tuning:
        logger.info("Starting hyperparameter tuning...")
        param_grid = {
            'lr': [0.001, 0.01],
            'batch_size': [32, 64, 128]
        }
        best_params, best_accuracy = hyperparameter_tuning(CNN, DEVICE, mnist_train_loader, mnist_test_loader, param_grid, epochs=3)
        logger.info(f"Best parameters: {best_params}, Best accuracy: {best_accuracy}")
        learning_rate = best_params['lr']
    else:
        learning_rate = args.learning_rate

    # Train base CNN on MNIST
    logger.info("Training base CNN on MNIST...")
    base_cnn = CNN().to(DEVICE)
    optimizer = optim.Adam(base_cnn.parameters(), lr=learning_rate)
    train_losses, test_losses, test_accuracies = train_and_evaluate(
        base_cnn, DEVICE, mnist_train_loader, mnist_test_loader, optimizer, args.mnist_epochs
    )

    # Visualizations for base CNN
    plot_learning_curves(train_losses, test_losses, test_accuracies, "Base CNN Learning Curves", OUTPUT_DIR)
    plot_weight_heatmaps(base_cnn, "Base CNN Weight Heatmaps", OUTPUT_DIR)
    visualize_feature_space(base_cnn, DEVICE, mnist_test_loader, "Base CNN MNIST Feature Space", OUTPUT_DIR)
    plot_confusion_matrix(base_cnn, DEVICE, mnist_test_loader, list(range(10)), "Base CNN MNIST Confusion Matrix", OUTPUT_DIR)

    # Save base CNN model
    torch.save(base_cnn.state_dict(), f"{OUTPUT_DIR}/base_cnn_mnist.pth")

    # Fine-tune on Fashion MNIST using different LoRA variants
    lora_variants = [
        ("StandardLoRA", StandardLoRALinear),
        ("ScaledLoRA", ScaledLoRALinear),
        ("MultiRankLoRA", lambda in_f, out_f: MultiRankLoRALinear(in_f, out_f, r=[4, 8]))
    ]

    lora_models = {}
    for lora_name, LoRAClass in lora_variants:
        logger.info(f"Fine-tuning with {lora_name} on Fashion MNIST...")
        
        if lora_name == "MultiRankLoRA":
            lora_model = create_lora_cnn(LoRAClass)
        else:
            lora_model = create_lora_cnn(LoRAClass, r=4, lora_alpha=1, lora_dropout=0.)
        
        lora_model = lora_model.to(DEVICE)
        lora_model.load_state_dict(base_cnn.state_dict(), strict=False)
        
        optimizer = optim.Adam(lora_model.parameters(), lr=learning_rate)
        
        lora_train_losses, lora_test_losses, lora_test_accuracies = [], [], []
        for epoch in range(args.fashion_mnist_epochs):
            train_loss = fine_tune(lora_model, DEVICE, fashion_train_loader, optimizer, epoch)
            test_loss, test_accuracy = train_and_evaluate(lora_model, DEVICE, fashion_train_loader, fashion_test_loader, optimizer, 1)
            lora_train_losses.append(train_loss)
            lora_test_losses.append(test_loss[0])
            lora_test_accuracies.append(test_accuracy[0])
        
        # Visualizations for LoRA model
        plot_learning_curves(lora_train_losses, lora_test_losses, lora_test_accuracies, f"{lora_name} Learning Curves", OUTPUT_DIR)
        plot_weight_heatmaps(lora_model, f"{lora_name} Weight Heatmaps", OUTPUT_DIR)
        plot_lora_weights(lora_model, f"{lora_name} LoRA Weights", OUTPUT_DIR)
        visualize_feature_space(lora_model, DEVICE, fashion_test_loader, f"{lora_name} Fashion MNIST Feature Space", OUTPUT_DIR)
        plot_confusion_matrix(lora_model, DEVICE, fashion_test_loader, list(range(10)), f"{lora_name} Fashion MNIST Confusion Matrix", OUTPUT_DIR)

        # Save LoRA model
        torch.save(lora_model.state_dict(), f"{OUTPUT_DIR}/{lora_name.lower()}_fashion_mnist.pth")
        
        lora_models[lora_name] = lora_model

    # Compare models
    all_models = {"Base CNN": base_cnn, **lora_models}
    comparison = compare_models(all_models, DEVICE, fashion_test_loader)
    
    logger.info("Model Comparison Results:")
    for model_name, results in comparison.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"Metrics: {results['metrics']}")
        logger.info(f"Total parameters: {results['total_params']}")
        logger.info(f"Trainable parameters: {results['trainable_params']}")

    # Grad-CAM visualization for a sample image
    sample_image, _ = next(iter(fashion_test_loader))
    for model_name, model in all_models.items():
        plot_grad_cam(model, DEVICE, sample_image[0], target_class=5, title=f"{model_name} Grad-CAM", output_dir=OUTPUT_DIR)

    logger.info("miniLora experiment completed successfully!")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
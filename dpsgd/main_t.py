import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from opacus import PrivacyEngine
from models import DenseModel, CNNModel
from utils import get_dataloaders, calculate_accuracy
import argparse

# --- CONFIGURATION ---
ROOT_DIR = Path(__file__).absolute().parent
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


BATCH_SIZE = 64
LEARNING_RATE = 0.01
L2NORM_BOUND = 4.0
SIGMA = 4.0
DATASET = 'mnist'  # 'mnist' ou 'cifar10'
MODEL_TYPE = 'dense' # 'dense' ou 'cnn'
USE_PRIVACY = True
PLOT_RESULTS = True
N_EPOCHS = 200
DELTA = 1e-5
MAX_EPS = 64.0

def plot_metrics(train_loss_scores, valid_loss_scores, train_acc_scores, valid_acc_scores, version):

        epochs_range = range(1, len(train_loss_scores) + 1)
        
        plt.figure(figsize=(8,6))
        plt.plot(epochs_range, train_loss_scores, color='blue', label='Training loss')
        plt.plot(epochs_range, valid_loss_scores, color='red', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(RESULTS_DIR / f"{version}-Loss-{N_EPOCHS}-{MODEL_TYPE}-{DATASET}.png")
        plt.close()
         
        plt.figure(figsize=(8,6))
        plt.plot(epochs_range, train_acc_scores, color='blue', label='Training accuracy')
        plt.plot(epochs_range, valid_acc_scores, color='red', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(RESULTS_DIR / f"{version}-Accuracy-{N_EPOCHS}-{MODEL_TYPE}-{DATASET}.png")
        plt.close()



# --- BOUCLE PRINCIPALE ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")

    train_loader, valid_loader, test_loader = get_dataloaders(DATASET, BATCH_SIZE)
    
    n_channels = 1 if DATASET == 'mnist' else 3
    image_size = 28 if DATASET == 'mnist' else 32
    input_dim = image_size * image_size * n_channels
    num_classes = 10

    if MODEL_TYPE == 'dense':
        model = DenseModel(input_dim, image_size*image_size, num_classes).to(device)
    else:
        model = CNNModel(n_channels, num_classes).to(device)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Intégration de la confidentialité 
    if USE_PRIVACY:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=SIGMA,
            max_grad_norm=L2NORM_BOUND,
        )

    train_loss_scores, valid_loss_scores = [], []
    train_acc_scores, valid_acc_scores = [], []

    start_time = time.time()
    should_terminate = False

    for epoch in range(1, N_EPOCHS + 1):
        if should_terminate:
            break
            
        print(f"\nEpoch {epoch}/{N_EPOCHS}")
        model.train()
        train_loss, train_acc = 0.0, 0.0
        
        for step, (images, targets) in enumerate(train_loader, 1):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += calculate_accuracy(outputs, targets)

            # Vérification du budget de confidentialité
            if USE_PRIVACY:
                epsilon = privacy_engine.get_epsilon(DELTA)
                if epsilon > MAX_EPS:
                    print(f"Budget dépassé: {epsilon:.4f} eps. Arrêt...")
                    should_terminate = True
                    break

        # Validation
        model.eval()
        valid_loss, valid_acc = 0.0, 0.0
        with torch.no_grad():
            for images, targets in valid_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
                valid_acc += calculate_accuracy(outputs, targets)

        # Moyennes des métriques
        train_loss_scores.append(train_loss / len(train_loader))
        train_acc_scores.append(train_acc / len(train_loader))
        valid_loss_scores.append(valid_loss / len(valid_loader))
        valid_acc_scores.append(valid_acc / len(valid_loader))

        time_taken = time.time() - start_time
        metrics_str = f"Train Loss: {train_loss_scores[-1]:.4f} - Train Acc: {train_acc_scores[-1]:.4f} | Valid Loss: {valid_loss_scores[-1]:.4f} - Valid Acc: {valid_acc_scores[-1]:.4f}"
        
        if USE_PRIVACY:
            print(f"{metrics_str} - spent eps: {epsilon:.4f} - time: {time_taken:.0f}s")
        else:
            print(f"{metrics_str} - time: {time_taken:.0f}s")

    # Test final
    model.eval()
    test_acc = 0.0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            test_acc += calculate_accuracy(outputs, targets)
    print(f"\nEntraînement terminé. Précision sur le Test Set: {test_acc/len(test_loader):.4f}")

    # Sauvegarde
    version = "DPSGD" if USE_PRIVACY else "SGD"
    torch.save(model.state_dict(), MODELS_DIR / f"{version}-{N_EPOCHS}-{MODEL_TYPE}-{DATASET}.pt")

    # Graphiques
    if PLOT_RESULTS:
        plot_metrics(train_loss_scores, valid_loss_scores, train_acc_scores, valid_acc_scores, version)
    
if __name__ == "__main__":
    main()
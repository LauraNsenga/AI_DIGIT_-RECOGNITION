import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pickle



def load_data(filename):
    """Load and normalize digit data from CSV"""
    print(f"Loading data from {filename}...")
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, :-1]  # All columns except last
    y = data[:, -1].astype(int)  # Last column as labels
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    # Reshape for CNN: (batch_size, channels=1, height=28, width=28)
    X = X.reshape(-1, 1, 28, 28)
    
    return X, y

def visualize_samples(X, y, n_samples=10):
    """Display sample digits from dataset"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('Sample Handwritten Digits', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        # X is already reshaped to (N, 1, 28, 28)
        img = X[i].squeeze()  # Remove channel dimension for display
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Label: {int(y[i])}', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_digits.png', dpi=150)
    print("✓ Saved sample visualizations to 'sample_digits.png'")
    plt.close()



class DigitCNN(nn.Module):
    """Convolutional Neural Network for digit recognition"""
    def __init__(self):
        super(DigitCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Regularization
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # Conv block 1: 28x28 -> 14x14
        x = self.pool(torch.relu(self.conv1(x)))
        
        # Conv block 2: 14x14 -> 7x7
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


def train_model(train_loader, val_loader, epochs=10, device='cpu'):
    """Train CNN with validation"""
    print(f"\n{'='*60}")
    print(f"Training CNN on: {device}")
    print(f"{'='*60}\n")
    
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_accuracies = []
    best_val_acc = 0
    
    for epoch in range(epochs):
      
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
       
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
      
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
      
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Loss: {avg_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Best: {best_val_acc:.2f}%")
    
    # =====  TRAINING  =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(train_losses, marker='o', linewidth=2)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(val_accuracies, marker='o', color='green', linewidth=2)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("\n✓ Saved training curves to 'training_curves.png'")
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}\n")
    
    return model, best_val_acc



def evaluate_model(model, test_loader, device='cpu'):
    """Comprehensive model evaluation with PyTorch"""
    print("\n📊 Evaluating model on test set...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # ===== CLASSIFICATION ====
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_predictions, digits=4))
    

    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("✓ Saved confusion matrix to 'confusion_matrix.png'")
    plt.close()
    

    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\n" + "="*60)
    print("PER-CLASS ACCURACY")
    print("="*60)
    for digit, acc in enumerate(per_class_acc):
        print(f"Digit {digit}: {acc:.2%}")
    

    accuracy = 100 * np.sum(all_predictions == all_labels) / len(all_labels)
    misclassified_idx = np.where(all_predictions != all_labels)[0]
    
    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {accuracy:.2f}%")
    print(f"Misclassified: {len(misclassified_idx)} out of {len(all_labels)}")
    print(f"{'='*60}\n")
    
    return accuracy, all_predictions, all_labels, misclassified_idx

def visualize_errors(X_test, y_test, predictions, misclassified, n_errors=10):
    """Visualize misclassified examples"""
    if len(misclassified) == 0:
        print("🎉 Perfect! No errors to visualize!")
        return
    
    n_show = min(n_errors, len(misclassified))
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle('Misclassified Examples', fontsize=16, fontweight='bold', color='red')
    
    for i, ax in enumerate(axes.flat):
        if i >= n_show:
            ax.axis('off')
            continue
        
        idx = misclassified[i]
        # X_test shape: (N, 1, 28, 28)
        img = X_test[idx].squeeze()  # Remove channel dimension
        true_label = int(y_test[idx])
        pred_label = int(predictions[idx])
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {true_label} | Pred: {pred_label}', 
                    fontsize=11, color='red', fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('misclassified_examples.png', dpi=150)
    print("✓ Saved misclassification analysis to 'misclassified_examples.png'")
    plt.close()



def main():
    print("\n" + "="*60)
    print("PYTORCH CNN DIGIT RECOGNITION SYSTEM")
    print("="*60 + "\n")
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name()}")
    else:
        print("ℹ️  Running on CPU (Training will be slower)")
    
    # ===== loading data =====
    training_file = input("\nEnter training data file (e.g., mnist_train.csv): ").strip()
    X_train_full, y_train_full = load_data(training_file)
    print(f"✓ Loaded {len(X_train_full)} training samples")
    
    # Visualize sample digits
    visualize_samples(X_train_full, y_train_full)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_full)
    y_train_tensor = torch.LongTensor(y_train_full)
    
    # Create dataset
    full_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    
    # Split into train and validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Training samples: {train_size}")
    print(f"✓ Validation samples: {val_size}")
    
    # ===== TRAIN OR LOAD MODEL =====
    user_input = input("\nTrain new model? (y/n): ").lower()
    
    if user_input == 'y':
        epochs = int(input("Number of epochs (default 10): ") or "10")
        model, best_val_acc = train_model(train_loader, val_loader, epochs, device)
        
        # Save model
        model_data = {
            'model_state_dict': model.state_dict(),
            'best_val_acc': best_val_acc,
            'architecture': 'CNN (2 conv layers, 128 FC units)',
            'training_samples': train_size
        }
        
        torch.save(model_data, 'digit_cnn_model.pth')
        print(f"✓ Model saved to 'digit_cnn_model.pth'")
    else:
        try:
            print("Loading saved model...")
            checkpoint = torch.load('digit_cnn_model.pth', map_location=device)
            model = DigitCNN().to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded model (Val Acc: {checkpoint['best_val_acc']:.2f}%)")
        except FileNotFoundError:
            print("❌ No saved model found. Training new model...")
            model, best_val_acc = train_model(train_loader, val_loader, 10, device)
    
    # ===== TEST MODEL =====
    test_file = input("\nEnter test data file (e.g., mnist_test.csv): ").strip()
    X_test, y_test = load_data(test_file)
    print(f"✓ Loaded {len(X_test)} test samples")
    
    # Create test dataloader
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate
    accuracy, predictions, labels, misclassified = evaluate_model(model, test_loader, device)
    
    # Visualize errors
    visualize_errors(X_test, y_test, predictions, misclassified)
    
    print(f"\n{'='*60}")
    print(f"🎯 FINAL TEST ACCURACY: {accuracy:.2f}%")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

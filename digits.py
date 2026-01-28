import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class DigitCNN(nn.Module):
    """Convolutional Neural Network for digit recognition"""
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_model(train_loader, val_loader, epochs=10):
    """Train CNN with validation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
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
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("✓ Saved training curves")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    print("\n📊 Evaluating model performance...")
    
    predictions = model.predict(X_test)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("✓ Saved confusion matrix to 'confusion_matrix.png'")
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class Accuracy:")
    for digit, acc in enumerate(per_class_acc):
        print(f"Digit {digit}: {acc:.2%}")
    
    # Overall metrics
    accuracy = np.sum(predictions == y_test) / len(y_test) * 100
    
    # Find misclassified examples
    misclassified = np.where(predictions != y_test)[0]
    print(f"\n✓ Total misclassified: {len(misclassified)} out of {len(y_test)}")
    
    return accuracy, predictions, misclassified

def visualize_errors(X_test, y_test, predictions, misclassified, n_errors=10):
    """Visualize misclassified examples"""
    if len(misclassified) == 0:
        print("No errors to visualize!")
        return
    
    n_show = min(n_errors, len(misclassified))
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('Misclassified Examples')
    
    for i, ax in enumerate(axes.flat):
        if i >= n_show:
            break
        
        idx = misclassified[i]
        img = X_test[idx].reshape(28, 28)
        true_label = int(y_test[idx])
        pred_label = int(predictions[idx])
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {true_label}, Pred: {pred_label}', color='red')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('misclassified_examples.png')
    print("✓ Saved misclassification analysis to 'misclassified_examples.png'")

def main():
    print("=" * 60)
    print("HANDWRITTEN DIGIT RECOGNITION SYSTEM")
    print("=" * 60)
    
    # Load training data
    training_file = input("\nEnter the training data file (e.g., mnist_train.csv): ").strip()
    X_train, y_train = load_data(training_file)
    print(f"✓ Loaded {len(X_train)} training samples")
    
    # Visualize samples
    visualize_samples(X_train, y_train)
    
    # Train or load model
    user_input = input("\nDo you want to train a new model? (y/n): ").lower()
    
    if user_input == 'y':
        model, best_arch, cv_score = train_model_with_validation(X_train, y_train)
        
        # Save model with metadata
        model_data = {
            'model': model,
            'architecture': best_arch,
            'cv_accuracy': cv_score,
            'training_samples': len(X_train)
        }
        
        with open('digit_classifier_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n✓ Model saved to 'digit_classifier_model.pkl'")
    else:
        try:
            with open('digit_classifier_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            model = model_data['model']
            print(f"✓ Loaded saved model (Architecture: {model_data['architecture']})")
        except FileNotFoundError:
            print("❌ No saved model found. Training new model...")
            model, best_arch, cv_score = train_model_with_validation(X_train, y_train)
    
    # Test model
    test_file = input("\nEnter the test data file (e.g., mnist_test.csv): ").strip()
    X_test, y_test = load_data(test_file)
    print(f"✓ Loaded {len(X_test)} test samples")
    
    # Evaluate
    accuracy, predictions, misclassified = evaluate_model(model, X_test, y_test)
    
    # Visualize errors
    visualize_errors(X_test, y_test, predictions, misclassified)
    
    print(f"\n{'=' * 60}")
    print(f"FINAL ACCURACY: {accuracy:.2f}%")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()

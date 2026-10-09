import torch


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    """Compute loss and collect labels and predictions."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / len(data_loader.dataset)
    return average_loss, all_labels, all_predictions

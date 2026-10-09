def train_one_epoch(model, data_loader, criterion, optimizer, device):
    """Train the model for one epoch and return loss and accuracy."""
    model.train()
    total_loss = 0.0
    total_correct = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()

    train_loss = total_loss / len(data_loader.dataset)
    train_accuracy = total_correct / len(data_loader.dataset)
    return train_loss, train_accuracy

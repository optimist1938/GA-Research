import torch
from tqdm import tqdm
from pose_3d.src.train_utils import grad_norm

def train(model, train_loader, test_init_loader, test_rot_loader, optimizer, scheduler, criterion, run, config):
    model.to(config.device)
    for _ in range(config.n_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(config.device), labels.to(config.device)

            # FOUR GOLDEN RULES
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        test_init_acc, test_rot_acc = test(model, test_init_loader, test_rot_loader, config)

        if run is not None:
            run.log({
                "train_loss": train_loss / len(train_loader),
                "train_acc": 100. * train_correct / train_total,
                "test_initial_acc": test_init_acc,
                "test_rotated_acc": test_rot_acc,
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm": grad_norm(model)
            })
        scheduler.step()

@torch.no_grad()
def test(model, initial_loader, rotated_loader, config):
    model.eval()
    accuracies = []
    for loader in [initial_loader, rotated_loader]:
        correct, total = 0, 0
        for images, labels in loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        accuracies.append(100. * correct / total)
    return accuracies[0], accuracies[1]
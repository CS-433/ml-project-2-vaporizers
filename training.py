import numpy as np
import torch
from torch.nn import *
import matplotlib.pyplot as plt


def train_epoch(model, dataloader, optimizer, criterion, scheduler, device, epoch):
    """ Train the model for one epoch.
    Args:
        model: model to train
        dataloader: loader of the dataset
        optimizer: optimizer
        criterion: loss function
        scheduler: scheduler
        device: device to use
        epoch: current epoch
    Returns:
        mean_loss: mean loss over the entire dataset
        mean_error: mean error over the entire dataset
    """

    model.train()

    mean_loss = 0
    mean_error = 0

    for batch_idx, data in enumerate(dataloader):
        inputs = data['feature']
        labels = data['label']
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Scheduler
        scheduler.step()

        # Save loss and error
        mean_loss += loss.item() * len(inputs)
        mean_error += torch.sum(torch.linalg.norm(labels - outputs, dim=1) / torch.linalg.norm(labels, dim=1)).item()

    mean_loss = mean_loss / len(dataloader.dataset)
    mean_error = mean_error / len(dataloader.dataset)

    if epoch % 100 == 0:
        print('Train epoch {}: loss = {}, error = {}, lr = {}'.format(epoch + 1, mean_loss, mean_error, scheduler.get_last_lr()[0]))

    return mean_loss, mean_error


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch):
    """ Validate the model.
    Args:
        model: model to validate
        dataloader: loader of the dataset
        criterion: loss function
        device: device to use
        epoch: current epoch
    Returns:
        loss: loss over the entire dataset
        error: error over the entire dataset
    """

    model.eval()

    loss = 0
    error = 0

    for batch_idx, data in enumerate(dataloader):
        inputs = data['feature']
        labels = data['label']
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss += criterion(outputs, labels).item() * len(inputs)
        error += torch.sum(torch.linalg.norm(labels - outputs, dim=1) / torch.linalg.norm(labels, dim=1)).item()

    loss = loss / len(dataloader.dataset)
    error = error / len(dataloader.dataset)

    if epoch % 100 == 0:
            print('Validation epoch {}: val_loss = {}, val_error = {}'.format(epoch + 1, loss, error))

    return loss, error


def run_training(train_gen, val_gen, num_epochs, model, lr, device='cpu'):
    """ Run the training.
    Args:
        train_gen: generator of the training set
        val_gen: generator of the validation set
        num_epochs: number of epochs
        model: model to train
        lr: learning rate
        device: device to use
    Returns:
        losses: list of training losses
        losses_val: list of validation losses
        errors: list of training errors
        errors_val: list of validation errors
    """

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.99)

    losses = []
    losses_val = []
    errors = []
    errors_val = []

    for epoch in range(num_epochs):
        loss, err = train_epoch(model, train_gen, optimizer, criterion, scheduler, device, epoch)
        losses.append(loss)
        errors.append(err)
        if val_gen != None:
            loss_val, err_val = validate(model, val_gen, criterion, device, epoch)
            losses_val.append(loss_val)
            errors_val.append(err_val)

    # Plot loss
    if val_gen != None:
        t = np.arange(1, num_epochs + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(t, losses)
        ax1.plot(t, losses_val)
        ax1.legend(['loss_train', 'loss_val'])
        ax2.plot(t, errors)
        ax2.plot(t, errors_val)
        ax2.legend(['err_train', 'err_val'])
        plt.show()
    else:
        t = np.arange(1, num_epochs + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(t, losses)
        ax1.legend(['loss_train'])
        ax2.plot(t, errors)
        ax2.legend(['err_train'])
        plt.show()

    return losses, losses_val, errors, errors_val


@torch.no_grad()
def compute_error(model, dataloader, device):
    """ Compute the error.
    Args:
        model: model to use
        dataloader: loader of the dataset
        device: device to use
    Returns:
        error: error over the entire dataset
    """

    model.eval()
    
    error = 0

    for batch_idx, data in enumerate(dataloader):
        inputs = data['feature']
        labels = data['label']
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        error += torch.sum(torch.linalg.norm(labels - outputs, dim=1) / torch.linalg.norm(labels, dim=1)).item()
    
    error = error / len(dataloader.dataset)

    return error


@torch.no_grad()
def predict(model, dataloader, device):
    """ Output the predictions of the trained model.
    Args:
        model: model to use
        dataloader: loader of the dataset, containing ONLY ONE BATCH
        device: device to use
    Returns:
        outputs: predictions of the model
    """

    model.eval()
    
    for batch_idx, data in enumerate(dataloader):
        inputs = data['feature']
        inputs = inputs.to(device)

        outputs = model(inputs)

    return outputs


def transform(v):
    """ Transform the labels.
    Args:
        v: labels
    Returns:
        v: transformed labels
    """
    return torch.sign(v) * torch.log(torch.abs(v) + 1)


def inv_transform(v):
    """ Inverse transform the labels.
    Args:   
        v: transformed labels
    Returns:
        v: labels
    """
    return torch.sign(v) * (torch.exp(torch.abs(v)) - 1)


def train_epoch_weighted(model, dataloader, optimizer, criterion, scheduler, sv, device, epoch):
    """ Train the model with the weighted loss for one epoch.
    Args:
        model: model to train
        dataloader: loader of the dataset
        optimizer: optimizer
        criterion: loss function
        scheduler: scheduler
        sv: singular values
        device: device to use
        epoch: current epoch
    Returns:
        mean_loss: mean loss over the entire dataset
        mean_error: mean error over the entire dataset
    """

    model.train()

    mean_loss = 0
    mean_error = 0

    for batch_idx, data in enumerate(dataloader):
        inputs = data['feature']
        labels = data['label']
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Transform labels
        labels = transform(labels)

        # Zero gradients
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels, sv)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Scheduler
        scheduler.step()

        # Anti-trasform labels and outputs
        labels = inv_transform(labels)
        outputs = inv_transform(outputs)

        # Save loss and error
        mean_loss += loss.item() * len(inputs)
        mean_error += torch.sum(torch.linalg.norm(labels - outputs, dim=1) / torch.linalg.norm(labels, dim=1)).item()

    mean_loss = mean_loss / len(dataloader.dataset)
    mean_error = mean_error / len(dataloader.dataset)

    if epoch % 100 == 0:
        print('Train epoch {}: loss = {}, error = {}, lr = {}'.format(epoch + 1, mean_loss, mean_error, scheduler.get_last_lr()[0]))

    return mean_loss, mean_error


@torch.no_grad()
def validate_weighted(model, dataloader, criterion, sv, device, epoch):
    """ Validate the model with the weighted loss.
    Args:
        model: model to validate
        dataloader: loader of the dataset
        criterion: loss function
        sv: singular values
        device: device to use
        epoch: current epoch
    Returns:
        loss: loss over the entire dataset
        error: error over the entire dataset
    """

    model.eval()

    loss = 0
    error = 0

    for batch_idx, data in enumerate(dataloader):
        inputs = data['feature']
        labels = data['label']
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Transform labels
        labels = transform(labels)

        outputs = model(inputs)

        loss += criterion(outputs, labels, sv).item() * len(inputs)

        # Anti-trasform labels and outputs
        labels = inv_transform(labels)
        outputs = inv_transform(outputs)

        error += torch.sum(torch.linalg.norm(labels - outputs, dim=1) / torch.linalg.norm(labels, dim=1)).item()

    loss = loss / len(dataloader.dataset)
    error = error / len(dataloader.dataset)

    if epoch % 100 == 0:
            print('Validation epoch {}: val_loss = {}, val_error = {}'.format(epoch + 1, loss, error))

    return loss, error


def run_training_weighted(train_gen, val_gen, num_epochs, model, sv, lr, device='cpu'):
    """ Run the training with the weighted loss.
    Args:
        train_gen: generator of the training set
        val_gen: generator of the validation set
        num_epochs: number of epochs
        model: model to train
        sv: singular values
        lr: learning rate
        device: device to use
    Returns:
        losses: list of training losses
        losses_val: list of validation losses
        errors: list of training errors
        errors_val: list of validation errors
    """

    model = model.to(device)
    sv = sv.to(device)

    def MSELossWeightedSV(outputs, labels, sv):
        # for each row, compute f(sv) * (outputs - labels) ** 2 (dot product)
        return torch.mean(torch.matmul(sv, (outputs - labels).T ** 2))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = MSELossWeightedSV
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.99)

    losses = []
    losses_val = []
    errors = []
    errors_val = []

    for epoch in range(num_epochs):
        loss, err = train_epoch_weighted(model, train_gen, optimizer, criterion, scheduler, sv, device, epoch)
        losses.append(loss)
        errors.append(err)
        if val_gen != None:
            loss_val, err_val = validate_weighted(model, val_gen, criterion, sv, device, epoch)
            losses_val.append(loss_val)
            errors_val.append(err_val)

    # Plot loss
    if val_gen != None:
        t = np.arange(1, num_epochs + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(t, losses)
        ax1.plot(t, losses_val)
        ax1.legend(['loss_train', 'loss_val'])
        ax2.plot(t, errors)
        ax2.plot(t, errors_val)
        ax2.legend(['err_train', 'err_val'])
        plt.show()
    else:
        t = np.arange(1, num_epochs + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(t, losses)
        ax1.legend(['loss_train'])
        ax2.plot(t, errors)
        ax2.legend(['err_train'])
        plt.show()

    return losses, losses_val, errors, errors_val


@torch.no_grad()
def compute_error_weighted(model, dataloader, device):
    """ Compute the error with the weighted loss.
    Args:
        model: model to use
        dataloader: loader of the dataset
        device: device to use
    Returns:
        error: error over the entire dataset
    """

    model.eval()
    
    error = 0

    for batch_idx, data in enumerate(dataloader):
        inputs = data['feature']
        labels = data['label']
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        # Anti-trasform outputs
        outputs = inv_transform(outputs)

        error += torch.sum(torch.linalg.norm(labels - outputs, dim=1) / torch.linalg.norm(labels, dim=1)).item()
        
    error = error / len(dataloader.dataset)

    return error


@torch.no_grad()
def predict_weighted(model, dataloader, device):
    """ Output the predictions of the trained model with the weighted loss.
    Args:
        model: model to use
        dataloader: loader of the dataset, containing ONLY ONE BATCH
        device: device to use
    Returns:
        outputs: predictions of the model
    """

    model.eval()
    
    for batch_idx, data in enumerate(dataloader):
        inputs = data['feature']
        inputs = inputs.to(device)

        outputs = model(inputs)

        # Anti-trasform outputs
        outputs = inv_transform(outputs)

    return outputs
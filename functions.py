import numpy as np
import itertools
import os
import torch
from torch.nn.functional import softmax
from sklearn import metrics
from IPython.display import clear_output
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

#Model Init    ----------------------------------------------------------------------------------  

class music_net(nn.Module):
  def __init__(self):
    # """Intitalize neural net layers"""
    super(music_net, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=0)
    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0)
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
    self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
    self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
    self.fc1 = nn.Linear(in_features=9856, out_features=10)

    self.batchnorm1 = nn.BatchNorm2d(num_features=8)
    self.batchnorm2 = nn.BatchNorm2d(num_features=16)
    self.batchnorm3 = nn.BatchNorm2d(num_features=32)
    self.batchnorm4 = nn.BatchNorm2d(num_features=64)
    self.batchnorm5 = nn.BatchNorm2d(num_features=128)

    self.dropout = nn.Dropout(p=0.3, inplace=False)
    
    # # Densenet init
    # self.densenet = models.densenet121()
    # self.densenet.eval()
    # num_ftrs = self.densenet.classifier.in_features
    # self.densenet.classifier = nn.Linear(num_ftrs, 10)
    

  def forward(self, x):
    # Conv layer 1.
    x = self.conv1(x)
    x = self.batchnorm1(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)

    # Conv layer 2.
    x = self.conv2(x)
    x = self.batchnorm2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)

    # Conv layer 3.
    x = self.conv3(x)
    x = self.batchnorm3(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)

    # Conv layer 4.
    x = self.conv4(x)
    x = self.batchnorm4(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)

    # Conv layer 5.
    x = self.conv5(x)
    x = self.batchnorm5(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)

    # Fully connected layer 1.
    x = torch.flatten(x, 1)
    x = self.dropout(x)
    x = self.fc1(x)
    x = F.softmax(x)
    
    # # Densenet layers
    # x = self.densenet(x)

    return x


#Plot Graphs    ----------------------------------------------------------------------------------  

def plot_loss_accuracy(train_loss, train_acc, train_auroc, validation_loss, validation_acc, validation_auroc):
  clear_output(wait=True)
  epochs = len(train_loss)
  _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 5.5))
  
  ax1.plot(list(range(epochs)), train_loss, label='Training Loss')
  ax1.plot(list(range(epochs)), validation_loss, label='Validation Loss')
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('Loss')
  ax1.set_title('Epoch vs Loss')
  ax1.legend()

  ax2.plot(list(range(epochs)), train_acc, label='Training Accuracy')
  ax2.plot(list(range(epochs)), validation_acc, label='Validation Accuracy')
  ax2.set_xlabel('Epochs')
  ax2.set_ylabel('Accuracy')
  ax2.set_title('Epoch vs Accuracy')
  ax2.legend()
  
  ax3.plot(list(range(epochs)), train_acc, label='Training AUROC')
  ax3.plot(list(range(epochs)), validation_acc, label='Validation AUROC')
  ax3.set_xlabel('Epochs')
  ax3.set_ylabel('AUROC')
  ax3.set_title('Epoch vs AUROC')
  ax3.legend()
  
  plt.show()

#Adaptive training     ----------------------------------------------------------------------------------  

def save_checkpoint(model, epoch, checkpoint_dir, stats):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "stats": stats,
    }

    filename = os.path.join(checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(epoch))
    torch.save(state, filename)


def restore_checkpoint(model, checkpoint_dir, cuda=False, force=False, pretrain=False):

    try:
        cp_files = [
            file_
            for file_ in os.listdir(checkpoint_dir)
            if file_.startswith("epoch=") and file_.endswith(".checkpoint.pth.tar")
        ]
    except FileNotFoundError:
        cp_files = None
        os.makedirs(checkpoint_dir)
    if not cp_files:
        print("No saved model parameters found")
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0, []

    # Find latest epoch
    for i in itertools.count(1):
        if "epoch={}.checkpoint.pth.tar".format(i) in cp_files:
            epoch = i
        else:
            break

    if not force:
        print(
            "Which epoch to load from? Choose in range [0, {}].".format(epoch),
            "Enter 0 to train from scratch.",
        )
        print(">> ", end="")
        inp_epoch = int(input())
        if inp_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0, []
    else:
        print("Which epoch to load from? Choose in range [1, {}].".format(epoch))
        inp_epoch = int(input())
        if inp_epoch not in range(1, epoch + 1):
            raise Exception("Invalid epoch number")

    filename = os.path.join(
        checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(inp_epoch)
    )

    print("Loading from checkpoint {}?".format(filename))

    if cuda:
        checkpoint = torch.load(filename)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

    try:
        start_epoch = checkpoint["epoch"]
        stats = checkpoint["stats"]
        if pretrain:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> Successfully restored checkpoint (trained for {} epochs)".format(
                checkpoint["epoch"]
            )
        )
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch, stats


def clear_checkpoint(checkpoint_dir):

    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")


def early_stopping(stats, curr_count_to_patience, global_min_loss):

    curr_loss = stats[-1][1]
    if curr_loss >= global_min_loss:
        curr_count_to_patience += 1
    else:
        global_min_loss = curr_loss
        curr_count_to_patience = 0
    return curr_count_to_patience, global_min_loss

#Getting Metrics    ----------------------------------------------------------------------------------  

def evaluate_epoch(
    axes,
    tr_loader,
    val_loader,
    te_loader,
    model,
    criterion,
    epoch,
    stats,
    include_test=False,
    update_plot=True,
):
    #Test training and validation sets

    def _get_metrics(loader):
        y_true, y_pred, y_score = [], [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in loader:
            with torch.no_grad():
                output = model(X)
                predicted = predictions(output.data)
                y_true.append(y)
                y_pred.append(predicted)
                y_score.append(softmax(output.data, dim=1))
                total += y.size(0)
                correct += (predicted == y).sum().item()
                running_loss.append(criterion(output, y).item())
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        y_score = torch.cat(y_score)
        loss = np.mean(running_loss)
        acc = correct / total
        auroc = metrics.roc_auc_score(y_true, y_score, multi_class="ovo")
        return acc, loss, auroc

    train_acc, train_loss, train_auc = _get_metrics(tr_loader)
    val_acc, val_loss, val_auc = _get_metrics(val_loader)

    stats_at_epoch = [
        val_acc,
        val_loss,
        val_auc,
        train_acc,
        train_loss,
        train_auc,
    ]
    if include_test:
        stats_at_epoch += list(_get_metrics(te_loader))

    stats.append(stats_at_epoch)
    utils.log_training(epoch, stats)

def predictions(logits):
    return logits.max(1)[1]

def save_cnn_training_plot():
    plt.savefig("cnn_training_plot.png", dpi=200)
    
    
def log_training(epoch, stats):
    splits = ["Validation", "Train", "Test"]
    metrics = ["Accuracy", "Loss", "AUROC"]
    print("Epoch {}".format(epoch))
    for j, split in enumerate(splits):
        for i, metric in enumerate(metrics):
            idx = len(metrics) * j + i
            if idx >= len(stats[-1]):
                continue
            print(f"\t{split} {metric}:{round(stats[-1][idx],4)}")
            

#Confusion Matrix Generation    ----------------------------------------------------------------------------------         
           
def gen_labels(loader, model):
    #returns true and predicted values
    y_true, y_pred = [], []
    for X, y in loader:
        with torch.no_grad():
            output = model(X)
            predicted = predictions(output.data)
            y_true = np.append(y_true, y.numpy())
            y_pred = np.append(y_pred, predicted.numpy())
    return y_true, y_pred


def plot_conf(loader, model, sem_labels, png_name):
    #Draw confusion matrix
    y_true, y_pred = gen_labels(loader, model)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues, interpolation="nearest")
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.set_label("Frequency", rotation=270, labelpad=10)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, z, ha="center", va="center")
    plt.gcf().text(0.02, 0.4, sem_labels, fontsize=9)
    plt.subplots_adjust(left=0.5)
    ax.set_xlabel("Predictions")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("True Labels")
    plt.savefig(png_name)

#Helper Functions    ----------------------------------------------------------------------------------  

def count_parameters(model):
    #Some interesting info
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#Train Function    ----------------------------------------------------------------------------------  

def train(model, device, tr_loader, va_loader, te_loader):\
    
    criterion =  nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    print("Number of float-valued parameters:", count_parameters(model))
    
    evaluate_epoch(
        axes, tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats
    )
    
    global_min_loss = stats[val_loss]
    
    #Set desired patience level
    patience = 5
    
    curr_count_to_patience = 0
    epoch = 0
    
    while curr_count_to_patience < patience:
        train_epoch(tr_loader, model, criterion, optimizer, device)
        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
            include_test=False,
        )
        save_checkpoint(model, epoch + 1, config("target.checkpoint"), stats)
        curr_count_to_patience, global_min_loss = early_stopping(
            stats, curr_count_to_patience, global_min_loss
        )
        
        epoch += 1
        
    print("Finished Training")
    
#Old train function
def train(model, device, train_loader, validation_loader, epochs):
  criterion =  nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
  train_loss, validation_loss = [], []
  train_acc, validation_acc = [], []
  
  for epoch in range(epochs):
    model.train()
    # keep track of the running loss
    running_loss = 0.
    correct, total = 0, 0

    for data, target in train_loader:
      # getting the training set
      data, target = data.to(device), target.to(device)
      # Get the model output (call the model with the data from this batch)
      output = model(data)
      # Zero the gradients out)
      optimizer.zero_grad()
      # Get the Loss
      loss  = criterion(output, target)
      # Calculate the gradients
      loss.backward()
      # Update the weights (using the training step of the optimizer)
      optimizer.step()

      running_loss += loss  # add the loss for this batch

      # get accuracy
      _, predicted = torch.max(output, 1)
      total += target.size(0)
      correct += (predicted == target).sum().item()

    # append the loss for this epoch
    train_loss.append(running_loss.detach().cpu().item()/len(train_loader))
    train_acc.append(correct/total)

    # evaluate on validation data
    model.eval()
    running_loss = 0.
    correct, total = 0, 0

    for data, target in validation_loader:
      # getting the validation set
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      output = model(data)
      loss = criterion(output, target)
      running_loss += loss.item()
      # get accuracy
      _, predicted = torch.max(output, 1)
      total += target.size(0)
      correct += (predicted == target).sum().item()

    validation_loss.append(running_loss/len(validation_loader))
    validation_acc.append(correct/total)
    plot_loss_accuracy(train_loss, train_acc, validation_loss, validation_acc)
  plot_loss_accuracy(train_loss, train_acc, validation_loss, validation_acc)
  plt.show()

  return train_loss, train_acc, validation_loss, validation_acc

def train_epoch(loader, model, criterion, optimizer, device):
    for data, target in loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      optimizer.zero_grad()
      loss  = criterion(output, target)
      loss.backward()
      optimizer.step()

    #   running_loss += loss 

    #   _, predicted = torch.max(output, 1)
    #   total += target.size(0)
    #   correct += (predicted == target).sum().item()
        
    


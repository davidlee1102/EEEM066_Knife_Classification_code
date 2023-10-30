## import libraries for training
import sys
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import torch.optim
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import knifeDataset
import timm
from utils import *

import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

import argparse

#####
import timm


class OwnModel(nn.Module):
    def __init__(self, config):
        super(OwnModel, self).__init__()
        self.our_model_1 = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True,
                                             num_classes=config.n_classes)
        self.our_model_2 = timm.create_model('resnet50', pretrained=True, num_classes=config.n_classes)

    def forward(self, x):
        # Forward pass through both models
        output1 = self.our_model_1(x)
        output2 = self.our_model_2(x)

        output = (output1 + output2) / 2
        return output


## My Own Model

class MyOwnModelRedesigned(nn.Module):
    def __init__(self, config):
        super(MyOwnModelRedesigned, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Second convolutional block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Third convolutional block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fifth convolutional block
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, config.n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.leaky_relu(self.fc4(x))
        x = self.fc3(x)
        return x


class MyOwnModel(nn.Module):
    def __init__(self, config):
        super(MyOwnModel, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Second convolutional block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Third convolutional block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fifth convolutional block
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 3 * 3, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, config.n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_classes', type=int, default=192)
    parser.add_argument('--img_width', type=int, default=224)
    parser.add_argument('--img_height', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.00005)
    parser.add_argument('--model_training', type=str, default='tf_efficientnet_b0')

    args = parser.parse_args()

    config.n_classes = args.n_classes
    config.img_weight = args.img_width
    config.img_height = args.img_height
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.learning_rate
    model_training = args.model_training

## Writing the loss and results
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")
log = Logger()
log.open("logs/%s_log_train.txt")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |----- Train -----|----- Valid----|---------|\n')
log.write('mode     iter     epoch    |       loss      |        mAP    | time    |\n')
log.write('-------------------------------------------------------------------------------------------\n')

train_epoch_arr, train_loss_arr, train_map_avg_arr, train_i_arr = [], [], [], []
val_epoch_arr, val_loss_arr, val_map_avg_arr, val_i_arr = [], [], [], []


## Training the model
def train(train_loader, model, criterion, optimizer, epoch, valid_accuracy, start):
    losses = AverageMeter()
    model.train()
    model.training = True
    for i, (images, target, fnames) in enumerate(train_loader):
        img = images.cuda(non_blocking=True)
        label = target.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            logits = model(img)
        loss = criterion(logits, label)
        losses.update(loss.item(), images.size(0))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        print('\r', end='', flush=True)
        message = '%s %5.1f %6.1f        |      %0.3f     |      %0.3f     | %s' % ( \
            "train", i, epoch, losses.avg, valid_accuracy[0], time_to_str((timer() - start), 'min'))
        print(message, end='', flush=True)
        train_epoch_arr.append(epoch)
        train_loss_arr.append(losses.avg)
        if isinstance(valid_accuracy[0], int):
            train_map_avg_arr.append(valid_accuracy[0])
        else:
            train_map_avg_arr.append(valid_accuracy[0].item())
        train_i_arr.append(i)
    log.write("\n")
    log.write(message)

    return [losses.avg]


# Validating the model

def evaluate(val_loader, model, criterion, epoch, train_loss, start):
    model.cuda()
    model.eval()
    model.training = False
    map = AverageMeter()
    with torch.no_grad():
        for i, (images, target, fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)

            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1)

            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5, img.size(0))
            print('\r', end='', flush=True)
            message = '%s   %5.1f %6.1f       |      %0.3f     |      %0.3f    | %s' % ( \
                "val", i, epoch, train_loss[0], map.avg, time_to_str((timer() - start), 'min'))
            print(message, end='', flush=True)
            # save the memories here
            val_epoch_arr.append(epoch)
            val_loss_arr.append(train_loss[0])
            val_map_avg_arr.append(map.avg.item())
            val_i_arr.append(i)

        log.write("\n")
        log.write(message)
    return [map.avg]


## Computing the mean average precision, accuracy
def map_accuracy(probs, truth, k=5):
    with torch.no_grad():
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        # top accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct / len(truth)

        accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
        map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
        acc1 = accs[0]
        acc5 = accs[1]
        return map5, acc1, acc5


######################## load file and get splits #############################
train_imlist = pd.read_csv("train.csv")
train_gen = knifeDataset(train_imlist, mode="train")
train_loader = DataLoader(train_gen, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=8)
val_imlist = pd.read_csv("test.csv")
val_gen = knifeDataset(val_imlist, mode="val")
val_loader = DataLoader(val_gen, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=8)

## Loading the model to run
print(config.n_classes)
print(config.img_weight)
print(config.batch_size)
if model_training == 'myown':
    model = MyOwnModel(config)
elif model_training == 'myownredesign':
    model = MyOwnModelRedesigned(config)
elif model_training != 'other':
    model = timm.create_model(model_training, pretrained=True, num_classes=config.n_classes)
else:
    model = OwnModel(config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
############################# Parameters #################################
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.epochs * len(train_loader), eta_min=0,
                                           last_epoch=-1)
criterion = nn.CrossEntropyLoss().cuda()

############################# Training #################################
start_epoch = 0
val_metrics = [0]
scaler = torch.cuda.amp.GradScaler()
start = timer()
# train
for epoch in range(0, config.epochs):
    lr = get_learning_rate(optimizer)
    train_metrics = train(train_loader, model, criterion, optimizer, epoch, val_metrics, start)
    val_metrics = evaluate(val_loader, model, criterion, epoch, train_metrics, start)
    ## Saving the model
    filename = "Knife-Effb0-E" + str(epoch + 1) + ".pt"
    torch.save(model.state_dict(), filename)

# history_list = [epoch_train_arr, train_loss_arr, map_avg_arr, i_arr]
df = pd.DataFrame(list(zip(train_epoch_arr, train_loss_arr, train_map_avg_arr, train_i_arr)),
                  columns=['Epochs', 'Training Loss', 'mAP', 'Iteration'])
df.to_csv("training_history.csv", index=False)

df = pd.DataFrame(list(zip(val_epoch_arr, val_loss_arr, val_map_avg_arr, val_i_arr)),
                  columns=['Epochs', 'Validation Loss', 'mAP', 'Iteration'])
df.to_csv("validation_history.csv", index=False)

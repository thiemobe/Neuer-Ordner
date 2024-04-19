import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

# Datensatz vorbereiten, reshaping, z-transformation (0-1), Pytorch tensor erstellen
data_Train = pd.read_csv('training.csv', delimiter=';')
data_Test = pd.read_csv('test.csv', delimiter=';')

training_length = len(data_Train)
test_length = len(data_Test)

num_folds = 5  # Anzahl der Folds für Cossvalidation
kfold = KFold(n_splits=num_folds, shuffle=True)


#####################################
# normalize all needed inputs (0,1) #
#####################################

sd_Train = np.reshape(np.array(data_Train['Schichtdicke']), (1, -1))
ll_Train = np.reshape(np.array(data_Train['Laserleistung']), (1, -1))
ss_Train = np.reshape(np.array(data_Train['Scangeschwindigkeit']), (1, -1))
hd_Train = np.reshape(np.array(data_Train['Hatchabstand']), (1, -1))
d_Train = np.reshape(np.array(data_Train['Dichte']), (1, -1))

sd_Test = np.reshape(np.array(data_Test['Schichtdicke']), (1, -1))
ll_Test = np.reshape(np.array(data_Test['Laserleistung']), (1, -1))
ss_Test = np.reshape(np.array(data_Test['Scangeschwindigkeit']), (1, -1))
hd_Test = np.reshape(np.array(data_Test['Hatchabstand']), (1, -1))
d_Test = np.reshape(np.array(data_Test['Dichte']), (1, -1))


### Definition von Minima, Maxima und Intervallgrenzen für die Skalierung
min_SD = 0.025; max_SD = 0.1
min_LL = 51; max_LL = 350
min_SS = 201; max_SS = 1700
min_HD = 0.05065; max_HD = 0.19939
min_D = 80; max_D = 100

low = 0; high = 1

normalized_SD_Train = (sd_Train-min_SD)/(max_SD-min_SD)*(high-low)+low
normalized_LL_Train = (ll_Train-min_LL)/(max_LL-min_LL)*(high-low)+low
normalized_SS_Train = (ss_Train-min_SS)/(max_SS-min_SS)*(high-low)+low
normalized_HD_Train = (hd_Train-min_HD)/(max_HD-min_HD)*(high-low)+low
normalized_D_Train = (d_Train-min_D)/(max_D-min_D)*(high-low)+low

normalized_SD_Test = (sd_Test-min_SD)/(max_SD-min_SD)*(high-low)+low
normalized_LL_Test = (ll_Test-min_LL)/(max_LL-min_LL)*(high-low)+low
normalized_SS_Test = (ss_Test-min_SS)/(max_SS-min_SS)*(high-low)+low
normalized_HD_Test = (hd_Test-min_HD)/(max_HD-min_HD)*(high-low)+low
normalized_D_Test = (d_Test-min_D)/(max_D-min_D)*(high-low)+low


#####################################################
# create tensor of features and target for ML Model #
#####################################################

sd_Train = torch.Tensor(normalized_SD_Train).to(torch.float).view(training_length, 1)
ll_Train = torch.Tensor(normalized_LL_Train).to(torch.float).view(training_length, 1)
ss_Train = torch.Tensor(normalized_SS_Train).to(torch.float).view(training_length, 1)
hd_Train = torch.Tensor(normalized_HD_Train).to(torch.float).view(training_length, 1)
d_Train = torch.Tensor(normalized_D_Train).to(torch.float).view(training_length, 1)


sd_Test = torch.Tensor(normalized_SD_Test).to(torch.float).view(test_length, 1)
ll_Test = torch.Tensor(normalized_LL_Test).to(torch.float).view(test_length, 1)
ss_Test = torch.Tensor(normalized_SS_Test).to(torch.float).view(test_length, 1)
hd_Test = torch.Tensor(normalized_HD_Test).to(torch.float).view(test_length, 1)
d_Test = torch.Tensor(normalized_D_Test).to(torch.float).view(test_length, 1)


x_Train = torch.cat((sd_Train, ll_Train, ss_Train, hd_Train), 1)
y_Train = d_Train


x_Test = torch.cat((sd_Test, ll_Test, ss_Test, hd_Test), 1)
y_Test = d_Test


##################
# create Dataset #
##################

class Densities_Train (Dataset):

    def __init__(self):
        self.x = x_Train
        self.y = y_Train
        self.n_samples = x_Train.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class Densities_Test (Dataset):

    def __init__(self):
        self.x = x_Test
        self.y = y_Test
        self.n_samples = x_Test.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset_Train = Densities_Train()
dataset_Test = Densities_Test()

for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset_Train)):
    train_dataset = torch.utils.data.Subset(dataset_Train, train_indices)

dataloader_train = DataLoader(dataset=dataset_Train, batch_size=10, shuffle=True)
dataloader_val = DataLoader(dataset=dataset_Test, batch_size=10, shuffle=False)

def Rescale(x):
    x = x.detach().numpy() * (max_D - min_D) + min_D
    return x

# Vorwärtsmodell definieren
class ForwardModel(nn.Module):
    def __init__(self):
        super(ForwardModel, self).__init__()
        self.linf1 = nn.Linear(4, 60)
        self.linf2 = nn.Linear(60, 40)
        self.linf3 = nn.Linear(40, 1)
        #self.batchnorm1 = nn.BatchNorm1d(20)
        #self.dropout1 = nn.Dropout(0.5)
        #self.batchnorm2 = nn.BatchNorm1d(20)
        #self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.linf1(x))
        x = torch.relu(self.linf2(x))
        x = torch.sigmoid(self.linf3(x))
        return x
    

# Modell erstellen
model = ForwardModel()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

epochs = 2000
loss_values = []    

for epoch in range(epochs):
    epoch_loss = 0.0  # Track the loss for the current epoch
    for batch_X, batch_y in dataloader_train:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        epoch_loss += loss.item()  # Accumulate the loss for the epoch
        model.zero_grad()
        loss.backward()
        optimizer.step()

    average_epoch_loss = epoch_loss / len(dataloader_train)  # Calculate average loss for the epoch
    loss_values.append(average_epoch_loss)  # Store the average loss

    print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {average_epoch_loss}')

# Plot the loss values
plt.plot(loss_values)
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

#######################################################################################################################
# Modell auf Testdaten evaluieren
model.eval()  # Setze das Modell in den Evaluationsmodus
with torch.no_grad():
    all_predictions = []
    all_targets = []
    
    #for batch_X, batch_y in dataloader_test:
    for batch_X, batch_y in dataloader_val:
        predictions = model(batch_X)
        predictions_rescaled = Rescale(predictions) #eingefügt
        targets_rescaled = Rescale(batch_y) #eingefügt
        all_predictions.append(predictions_rescaled)
        all_targets.append(targets_rescaled)
        #all_targets.append(batch_y)


# Kombiniere alle Vorhersagen und Ziele
#all_predictions = torch.cat(all_predictions).numpy()
#all_targets = torch.cat(all_targets).numpy()
#all_predictions = torch.cat(all_predictions)
#all_targets = torch.cat(all_targets)
all_predictions = np.concatenate(all_predictions)
all_targets = np.concatenate(all_targets)

model.eval()  # Setze das Modell in den Evaluationsmodus
with torch.no_grad():
    all_train_predictions = []
    all_train_targets = []

    for batch_X, batch_y in dataloader_train:
        train_predictions = model(batch_X)
        train_predictions_rescaled = Rescale(train_predictions)  # eingefügt
        train_targets_rescaled = Rescale(batch_y)  # eingefügt
        all_train_predictions.append(train_predictions_rescaled)
        all_train_targets.append(train_targets_rescaled)

# Kombiniere alle Vorhersagen und Ziele für Trainingsdaten
all_train_predictions = np.concatenate(all_train_predictions)
all_train_targets = np.concatenate(all_train_targets)

# Daten plotten
plt.scatter(all_targets, all_predictions, alpha=0.5, label='Testdaten')
plt.scatter(all_train_targets, all_train_predictions, alpha=0.5, label='Trainingsdaten')  #hinzugefügt
plt.title('Scatterplot der Vorhersagen vs. tatsächliche Werte')
plt.xlabel('Tatsächliche Werte')
plt.ylabel('Vorhersagen')
plt.legend()
plt.show()

# Calculate Mean Squared Error (MSE) for validation data
mse_val = mean_squared_error(all_targets, all_predictions)
print(f'Mean Squared Error (Validation): {mse_val}')
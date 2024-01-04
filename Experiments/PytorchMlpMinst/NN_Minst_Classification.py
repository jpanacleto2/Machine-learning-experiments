import time
import torch
from PIL import Image
from torch import nn,save,load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import numpy as np
from sklearn import metrics

args = {
    'batch_size': 5,
    'num_workers': 4,
    'num_classes': 10,
    'lr': 1e-4,
    'weight_decay': 5e-4,
    'num_epochs': 10
}

train_set = datasets.MNIST('Projects/MlpMinst/data', 
                           train=True, 
                           transform=transforms.ToTensor(),
                           download=True)


train_loader = DataLoader(train_set, 
                          batch_size=args['batch_size'], 
                          shuffle=True, 
                          num_workers=args['num_workers'])

class ImageClasssifier(nn.Module):

  def __init__(self, input_size, hidden_size, out_size):
    super(ImageClasssifier, self).__init__()

    self.features  = nn.Sequential(
                      nn.Linear(input_size, hidden_size),
                      nn.ReLU(),
                      nn.Linear(hidden_size, hidden_size),
                      nn.ReLU()
                    )
    self.out     = nn.Linear(hidden_size, out_size)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, X):
    
    X = X.view(X.size(0), -1)

    feature = self.features(X)
    output  = self.softmax(self.out(feature))

    return output

input_size  = 28 * 28
hidden_size = 128
out_size    = 10 #classes

torch.manual_seed(42)
clf = ImageClasssifier(input_size, hidden_size, out_size).to('cpu') #cast na CPU

## Loss
criterion = nn.CrossEntropyLoss().to('cpu')

## Otimizador
opt = Adam(clf.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

# Funcao de treino
if __name__ == "__main__":
    for epoch in range(args['num_epochs']):
        start = time.time()

        epoch_loss = []

        for batch in train_loader:
            dado, rotulo = batch

            dado = dado.to('cpu') # cpu cast
            rotulo = rotulo.to('cpu') #cpu cast

            pred = clf(dado)
            loss = criterion(pred, rotulo)
            epoch_loss.append(loss.cpu().data)

            loss.backward()
            opt.step()

        epoch_loss = np.asarray(epoch_loss)
        end = time.time()


        print("Epoch %d, Loss: %.4f +/- %.4f, Time: %.2f" % (epoch, epoch_loss.mean(), epoch_loss.std(), end - start))

    with open('Projects/MlpMinst/model_state.pt', 'wb') as f: 
        save(clf.state_dict(), f) 

    with open('Projects/MlpMinst/model_state.pt', 'rb') as f: 
        clf.load_state_dict(load(f))  

    img = Image.open('Projects/MlpMinst/images/img_1.jpg') 
    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')

    print(torch.argmax(clf(img_tensor)))
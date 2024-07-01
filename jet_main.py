import torch.nn as nn
import torch
import torch.optim as optim

from jet_dataset import Segm_ds
from jet_model import JET

from torchvision import transforms
from torch.utils.data import DataLoader

from jet_utils import save_chkpt, chk_accr, train_f
# from  monai.losses import DiceLoss

import wandb


LEARNING_RATE = 1e-4
BATCH_SIZE = 4
LOAD_CHEKPT = False
SAVE_CHEKPT = True
EPOCHS = 5

if __name__ == "__main__":

    # wandb.login(key='8e7aa7fe4ddfe9a81bfda79a06aef00e7729417d')
    # wandb.init(project='jet_fundus')

    # wandb.config.update({
    # "learning_rate": LEARNING_RATE,
    # "epochs": EPOCHS,
    # "batch_size": BATCH_SIZE
    # })

    trf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    ])

    mask_trf = transforms.Compose([
        transforms.ToTensor()
    ])


    train_ds = DataLoader(Segm_ds(tt = 'train',trf = trf, mask_trf = mask_trf), batch_size=BATCH_SIZE, shuffle=True)
    test_ds = DataLoader(Segm_ds(tt = 'test',trf = trf, mask_trf = mask_trf), batch_size=BATCH_SIZE, shuffle=True)

    jet = JET(3, 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    jet.to(device)

    criterion = nn.BCEWithLogitsLoss()

    if LOAD_CHEKPT:
        chkpt=torch.load('models/jet_new0.pth.tar')
        jet.load_state_dict(chkpt['state_dict'])

    optimizer = optim.Adam(jet.parameters(), lr = LEARNING_RATE)

    scaler = torch.cuda.amp.GradScaler()

    print('\ntraining started')
    # wandb.watch(jet, log='all', log_freq=100)


    for i in range(EPOCHS):
        running_loss = train_f(loader=train_ds, criterion=criterion, optimizer=optimizer, scaler=scaler, model=jet)

        chkpt = {
        'state_dict': jet.state_dict(),
        'optimizer': optimizer.state_dict()
        }

    #     wandb.log({
    #     "train_loss": running_loss,
    # }, step=i + 1)
        
        print(f"epoch {i + 1}, running loss: {running_loss}")

        if SAVE_CHEKPT: save_chkpt(chkpt)

        chk_accr(loader=test_ds, model=jet, device=device)

    wandb.finish()
from tqdm import tqdm
import torch 
import torchvision
# from monai.losses import DiceLoss

# class ModifLoss(torch.nn.Module):
#     def __init__(self, w_bce = 0.7, w_dce = 0.3) -> None:
#         super().__init__()
#         self.dice = DiceLoss()
#         self.Bcelgt = torch.nn.BCEWithLogitsLoss()
#         self.w_bce = w_bce
#         self.w_dce = w_dce
#     def __call__(self,y_true: torch.Any, y_pred: torch.Any) -> torch.Any:
#         dce = self.dice(y_pred, y_true)
#         bce = self.Bcelgt(y_pred, y_true)
#         return self.w_bce * bce + self.w_dce * dce
        

def save_chkpt(state, path='models/jet_new0.pth.tar'):
    print('chkpoint saved')
    torch.save(state, path)


def train_f(loader, criterion, optimizer, scaler, model):
    loop = tqdm(loader)
    device = 'cuda'
    running_loss = 0.0

    for batch_idx, (inps, label) in enumerate(loop):
        inps = inps.to(device)
        label = label.float().to(device)

        with torch.cuda.amp.autocast():
            preds = model(inps)
            loss = criterion(preds, label)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
        running_loss += loss.item()
    
    return running_loss / len(loader)


def chk_accr(loader, model, device='cuda'):
    num_crct = 0.0
    num_pix = 0.0

    dice_scr = 0.0
    TP = 0.0
    FP = 0.0
    FN = 0.0
    IoU = 0.0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            num_crct += (preds == y).sum().item()
            num_pix += torch.numel(preds)
            
            dice_scr += (2 * (preds * y).sum().item()) / ((preds + y).sum().item() + 1e-8)
            
            TP += ((preds == 1) & (y == 1)).sum().item()
            FP += ((preds == 1) & (y == 0)).sum().item()
            FN += ((preds == 0) & (y == 1)).sum().item()
            
            IoU += (TP / (TP + FP + FN + 1e-8))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    
    print(f"{num_crct}/{num_pix}, acc {num_crct/num_pix * 100:.2f}")
    print(f"dice scr: {dice_scr/len(loader)}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"F-score: {(2 * precision * recall)/(precision + recall):.4f}")
    print(f"IoU: {IoU/len(loader):.4f}")



def chk_imgs(loader, count, model, path='imgs/', device='cuda'):
    print('ok')
    model.eval()
    cnt = 0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            torchvision.utils.save_image(preds, f"{path}ppred_{cnt}.png")
            if preds.shape != y.shape:
                y = y.unsqueeze(1)
            torchvision.utils.save_image(y, f"{path}llabel_{cnt}.png")
            cnt += 1
            if cnt > count: break

    
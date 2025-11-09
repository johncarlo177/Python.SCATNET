import torch, torch.optim as optim
from torch.utils.data import DataLoader
from src.models.scatnet import SCATNet
from src.data.dataset_loader import HSIRandomDataset
from src.training.losses import cb_focal_loss

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SCATNet(in_bands=64, num_classes=5).to(device)
    ds = HSIRandomDataset(n=400, bands=64, H=16, W=16, classes=5)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(2):
        tot=0; n=0
        for x,y in dl:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = cb_focal_loss(logits, y)
            loss.backward(); opt.step()
            tot += loss.item()*x.size(0); n += x.size(0)
        print(f'Epoch {epoch+1}: loss={tot/n:.4f}')
    torch.save(model.state_dict(), 'outputs/scatnet_toy.pth')

if __name__ == '__main__': main()

import torch.nn as nn
import torch


class Jet_Dwn(nn.Module):
    def __init__(self, inn, out) -> None:
        super(Jet_Dwn, self).__init__()
        self.jet_subconv = nn.Sequential(
            nn.Conv2d(in_channels = inn, out_channels = out, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = out, out_channels = out, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )
    def forward(self, inp):
        return self.jet_subconv(inp)


class Jet_Up(nn.Module):
    def __init__(self, inn, out) -> None:
        super(Jet_Up, self).__init__()
        self.jet_subup = nn.ConvTranspose2d(in_channels = inn, out_channels = inn//2, kernel_size=2, stride=2)
        self.jet_subconv4up = Jet_Dwn(inn = inn, out = out)

    def forward(self, inp, prev_op):
        temp = self.jet_subup(inp)
        temp = torch.cat((temp, prev_op), dim = 1)
        return self.jet_subconv4up(temp)
    

class JET(nn.Module):
    def __init__(self, inn, out) -> None:
        super(JET, self).__init__()

        self.d = nn.ModuleList()
        self.d.extend([Jet_Dwn(inn = inn, out = 16), 
                       Jet_Dwn(inn = 16, out = 32),
                       Jet_Dwn(inn = 32, out = 64),
                       Jet_Dwn(inn = 64, out = 128)])
        
        self.nek = Jet_Dwn(inn = 128, out = 256)

        self.u = nn.ModuleList()
        self.u.extend([Jet_Up(inn = 256, out = 128),
                       Jet_Up(inn = 128, out = 64),
                       Jet_Up(inn = 64, out = 32),
                       Jet_Up(inn = 32, out = 16)])
 
        self.ot = nn.Conv2d(in_channels = 16, out_channels=out, kernel_size=1, padding='same')

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, inp):

        skip_cons = []

        for d in self.d:
            skip_cons.append(d(inp))
            inp = self.pool(skip_cons[-1])
        
        inp  = self.nek(inp)

        skip_cons = skip_cons[::-1]

        for i in range(len(self.u)):
            inp = self.u[i](inp, skip_cons[i])

        return self.ot(inp)
"""Analytic temperature-rise model fitted to the chip-midplane label grid."""
import torch
from torch import nn

class AnalyticThermalModel(nn.Module):
    def __init__(self, width, height, num_chiplets, num_grid_x, num_grid_y):
        super().__init__()
        self.width = width
        self.height = height
        self.num_chiplets = num_chiplets
        xgrid = (torch.arange(num_grid_x) + 0.5) / num_grid_x * width
        ygrid = (torch.arange(num_grid_y) + 0.5) / num_grid_y * height
        xgrid, ygrid = torch.meshgrid(xgrid, ygrid, indexing="ij")
        # meshgrid returns overlapping views; independent storage is needed
        # for load_state_dict() to copy checkpoint buffers into this model.
        self.register_buffer("xgrid", xgrid[None, None].clone())
        self.register_buffer("ygrid", ygrid[None, None].clone())
        self.amp = nn.Parameter(torch.ones(1) * 1e3)
        self.bias = nn.Parameter(torch.zeros(1))
        self.heff = nn.Parameter(torch.ones(1))
        self.decay = nn.Parameter(torch.ones(1, num_chiplets, 1, 2))

    def forward(self, input_data):
        x, y, length, width, power = input_data
        chips = self.num_chiplets
        batch = x.shape[0]
        xc = x.view(-1, chips, 1, 1)
        yc = y.view(-1, chips, 1, 1)
        lc = length.view(-1, chips, 1, 1)
        wc = width.view(-1, chips, 1, 1)
        xgrid = self.xgrid.expand(batch, chips, -1, -1)
        ygrid = self.ygrid.expand(batch, chips, -1, -1)
        power = power.to(x.device).reshape(-1, chips, 1, 1)
        val = self._main_term(xgrid - xc, ygrid - yc, lc, wc)
        return (power * (self.amp * val + self.bias)).sum(dim=1, keepdim=True)

    def _main_term(self, xdist, ydist, length, width):
        decay = self.decay
        ax = decay[..., :1]
        ay = decay[..., 1:2]
        val = (
            self._fabc(self.heff, ax * (length / 2 - xdist), ay * (width / 2 - ydist))
            + self._fabc(self.heff, ax * (length / 2 - xdist), ay * (width / 2 + ydist))
            + self._fabc(self.heff, ax * (length / 2 + xdist), ay * (width / 2 - ydist))
            + self._fabc(self.heff, ax * (length / 2 + xdist), ay * (width / 2 + ydist))
        )
        return val / length / width

    @staticmethod
    def _fabc(a, b, c):
        a = a.double()
        b = b.double()
        c = c.double()
        delta = torch.sqrt(a**2 + b**2 + c**2)
        val = (
            b * torch.log((c + delta) / (a**2 + b**2) ** 0.5)
            + c * torch.log((b + delta) / (a**2 + c**2) ** 0.5)
            - a * torch.arctan(b * c / a / delta)
        )
        return val.float()

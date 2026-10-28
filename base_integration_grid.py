import torch
from torch import nn

class SpatialIntegrationGrid(nn.Module):

    def __init__(self, field, cache, kind='legacy'):
        super().__init__()
        self.kind = kind
        self.cell_size = 0.6
        if kind not in ('legacy', 'cells', 'contrast32'):
            raise ValueError('Unknown spatial integration grid')
        if kind == 'legacy':
            xy = field.xy.detach().clone()
            area = torch.ones(len(xy))
            variance = torch.full((len(xy),), 0.6 ** 2 / 12)
            parent = torch.arange(len(xy))
            self.selected = []
        else:
            cells = torch.as_tensor(cache['cells'], dtype=torch.float32)
            centres = (cells + 0.5) * self.cell_size
            parent = torch.arange(len(cells))
            xy = centres
            area = torch.ones(len(xy))
            variance = torch.full((len(xy),), 0.6 ** 2 / 12)
            self.selected = []
            if kind == 'contrast32':
                offsets = torch.tensor([[-0.15, -0.15], [-0.15, 0.15], [0.15, -0.15], [0.15, 0.15]])
                children = centres[:, None] + offsets
                with torch.no_grad():
                    values = field.source(children)['mean']
                    noise = field.source(centres)['variance'].clamp_min(0.25)
                    contrast = (values.var(1, unbiased=False) / noise).sum(-1)
                    selected = contrast.topk(min(32, len(cells))).indices
                keep = torch.ones(len(cells), dtype=torch.bool)
                keep[selected] = False
                xy = torch.cat((centres[keep], children[selected].reshape(-1, 2)))
                area = torch.cat((torch.ones(int(keep.sum())), torch.full((4 * len(selected),), 0.25)))
                variance = torch.cat((torch.full((int(keep.sum()),), 0.6 ** 2 / 12), torch.full((4 * len(selected),), 0.3 ** 2 / 12)))
                parent = torch.cat((parent[keep], selected.repeat_interleave(4)))
                self.selected = selected.tolist()
        for name, value in [('xy', xy), ('log_area', area.log()), ('variance', variance), ('parent', parent)]:
            self.register_buffer(name, value, persistent=False)

    def description(self):
        return dict(kind=self.kind, nodes=len(self.xy), source_cell_size_m=self.cell_size, selected_source_cells=self.selected, total_relative_area=float(self.log_area.exp().sum()))

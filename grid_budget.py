from pathlib import Path
import numpy as np

def select_indices(xy, count):
    xy = np.asarray(xy, dtype=float)
    if count >= len(xy):
        return np.arange(len(xy))
    if count < 2:
        raise ValueError('Full M3 needs at least two spatial candidates')
    first = int(np.lexsort((xy[:, 1], xy[:, 0]))[0])
    selected = [first]
    nearest = np.square(xy - xy[first]).sum(-1)
    while len(selected) < count:
        nearest[selected] = -1.0
        following = int(np.argmax(nearest))
        selected.append(following)
        nearest = np.minimum(nearest, np.square(xy - xy[following]).sum(-1))
    return np.asarray(sorted(selected), dtype=int)

def reduce_readout(work, count):
    if count >= 220:
        return []
    code = Path(work) / 'anchor_counts/B01_B02_B03_B04/code'
    reports = []
    for head in sorted((code / 'potentials_v5/original_fixed20').glob('*/head.npz')):
        with np.load(head, allow_pickle=False) as z:
            data = {k: z[k].copy() for k in z.files}
        original = len(data['grid_xy'])
        index = select_indices(data['grid_xy'], count)
        for key in ['potential', 'gaussian', 'correction', 'predicted_rssi_grid']:
            data[key] = data[key][:, index]
        data['grid_xy'] = data['grid_xy'][index]
        np.savez_compressed(head, **data)
        cavity = head.parent / 'wifi_CS_cavity.npz'
        with np.load(cavity, allow_pickle=False) as z:
            data = {k: z[k].copy() for k in z.files}
        cavity_index = select_indices(data['grid_xy'], count)
        data['grid_xy'] = data['grid_xy'][cavity_index]
        data['log_probability'] = data['log_probability'][:, cavity_index]
        np.savez_compressed(cavity, **data)
        reports.append(dict(trajectory_id=head.parent.name, original_nodes=original, M3_nodes=len(index), indices=index.tolist(), upstream_motion_channels=220, reference_map_nodes=220))
    if not reports:
        raise ValueError('No Full M3 readout to reduce')
    return reports

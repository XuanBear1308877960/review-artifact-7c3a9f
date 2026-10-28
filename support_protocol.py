from collections import Counter
PROTOCOL = 'fixed20_v1'
EXCLUDED_TEST_TRAJECTORIES = ('redmik40s_block00_seg003', 'redmik40s_block00_seg008', 'redmik40s_block01_seg010', 'redmik40s_block01_seg011')

def selected_segments(manifest, partition, cfg):
    segments = [s for s in manifest['segments'] if s['full_length']]
    protocol = cfg.get('evaluation', {}).get('protocol', PROTOCOL)
    if protocol not in (PROTOCOL, 'full24_reference'):
        raise ValueError('Unknown evaluation protocol: ' + str(protocol))
    if partition == 'test' and protocol == PROTOCOL:
        ids = {s['trajectory_id'] for s in segments}
        if not set(EXCLUDED_TEST_TRAJECTORIES).issubset(ids):
            raise ValueError('fixed20_v1 requires the original test manifest; do not prefilter or change its IDs')
        segments = [s for s in segments if s['trajectory_id'] not in EXCLUDED_TEST_TRAJECTORIES]
        if Counter((s['device'] for s in segments)) != dict(redmik40s=7, realme2=13):
            raise ValueError('fixed20_v1 must contain 7 K40S and 13 realme2 trajectories')
    return segments

def protocol_metadata(manifest, partition, cfg):
    segments = selected_segments(manifest, partition, cfg)
    protocol = cfg.get('evaluation', {}).get('protocol', PROTOCOL)
    return dict(name=protocol, partition=partition, selection_date='2026-10-14', interpretation='Frozen development subset selected after error inspection' if protocol == PROTOCOL else 'Original full-set reference', excluded_trajectory_ids=list(EXCLUDED_TEST_TRAJECTORIES) if partition == 'test' and protocol == PROTOCOL else [], included_trajectory_ids=[s['trajectory_id'] for s in segments], trajectory_count=len(segments), devices=dict(Counter((s['device'] for s in segments))))

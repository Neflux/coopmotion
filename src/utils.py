def detailed_name(g):
    targets = ['N', 'number_of_samples', 'eps', 'run', 'net']
    final = {}
    for k, v in g.items():
        if k in targets:
            if k == 'net':
                final[k] = type(v).__name__
            elif k == 'run':
                final['t'] = f"\b\b{'holo-' if v.task.holonomic else 'nonholo-'}{type(v.task).__name__}"
                sense_info = {'range': v.sensor.range, 'subs': v.sensor.subset}
                for k2, v2 in sense_info.items():
                    if v2 is not None:
                        final[k2] = v2
                if v.sensor.sorted:
                    final['sorted'] = '\b'
            elif k == 'number_of_samples':
                final['n'] = v
            else:
                final[k] = v

    final = sorted(final.items(), key=lambda x: x[0].lower())
    return '_'.join([f"{k.replace('_', '-')}={v}" for k, v in final])

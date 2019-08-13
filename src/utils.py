def detailed_name(g):
    targets = ['N', 'number_of_samples', 'eps', 'run', 'net']
    final = {}
    for k, v in g.items():
        if k in targets:
            if k == 'net':
                final[k] = type(v).__name__
            elif k == 'run':
                final['t'] = type(v.task).__name__
                mode = v.sensor.__qualname__[6:v.sensor.__qualname__.find('.')]
                final['s'] = f"{mode}-{v.sensor.get_params()}"
            elif k == 'number_of_samples':
                final['n'] = v
            else:
                final[k] = v

    final = sorted(final.items(), key=lambda x: x[0].lower())
    return '_'.join([f"{k.replace('_', '-')}={v}" for k, v in final])

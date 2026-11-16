#!/usr/bin/env python3
# grain-pick.py <grain.csv>: best nb (lowest mean ms, ok runs) per problem -> shell exports
import csv, sys, collections
best = {}
for x in csv.DictReader(open(sys.argv[1])):
    if x['status'] != 'ok' or not x['avg_ms']: continue
    k = (x['app'], x['variant'], int(x['size'])); v = float(x['avg_ms'])
    if k not in best or v < best[k][1]: best[k] = (int(x['nb']), v)
def g(app, var, sizes, dflt):
    return ','.join(str(best.get((app, var, s), (d, 0))[0]) for s, d in zip(sizes, dflt))
out = {'CLG': g('cloverleaf', '', [64, 256, 1024], [1, 8, 2]),
       'SWG': g('sw4lite', '', [16, 32, 64], [2, 2, 1]),
       'J2G': g('polybench', 'jacobi-2d', [64, 256, 1024], [2, 16, 64]),
       'H3G': g('polybench', 'heat-3d', [32, 64, 128], [8, 32, 64]),
       'FDG': g('polybench', 'fdtd-2d', [64, 256, 1024], [1, 8, 32]),
       'J1G': g('polybench', 'jacobi-1d', [1024, 65536, 1048576], [1, 1, 1]),
       'MWG': g('miniweather', '', [50, 200, 800], [2, 4, 8])}
for k, v in out.items(): print('export %s=%s' % (k, v))

import itertools
import sys

c = int(sys.argv[1])
for e in itertools.combinations(sys.argv[2:], c):
    print(' '.join(e))

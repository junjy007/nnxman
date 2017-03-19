"""Try

Usage:
  try02_parsearg.py <name> [--opta=<a> [--optb=<b>]]

"""
from docopt import docopt
if __name__=='__main__':
    import sys
    print sys.argv
    arg=docopt(__doc__, version="1.0")
    print arg

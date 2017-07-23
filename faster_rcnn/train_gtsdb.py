import sys
import os.path

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + '/..')

from lib.datasets.factory import get_imdb

imdb = get_imdb('gtsdb_train')

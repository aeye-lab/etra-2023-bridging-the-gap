import sys
import os

# this includes '../source/' to syspath to easily import project modules from notebooks directory
project_src_path = os.path.abspath(
    os.path.join(os.pardir, 'source'))
if project_src_path not in sys.path:
    sys.path.append(project_src_path)

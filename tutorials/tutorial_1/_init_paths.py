###########################################################################################
#                                                                                         #
# Set up paths                                           #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
###########################################################################################

import sys
import os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

currentPath = os.path.dirname(os.path.realpath(__file__))

# Add common to PYTHONPATH
libPath = os.path.join(currentPath, '..', 'common')
add_path(libPath)
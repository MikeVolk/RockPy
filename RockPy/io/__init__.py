import os
import pkgutil
from RockPy import installation_directory
__all__ = []
for loader, module_name, is_pkg in pkgutil.walk_packages([installation_directory+'/io']):
    print(module_name)
    __all__.append(module_name)
    module = loader.find_module(module_name).load_module(module_name)
    exec('%s = module' % module_name)
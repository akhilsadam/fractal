import hashlib
import importlib.util
import inspect
import re 
import os 

def consistent(d1,d2):
    for k in d2.keys():
        if 'train' in k or 'test' in k:
            continue
        assert d1.__dict__[k] == d2[k], f'{k} mismatch: {d1.__dict__[k]} != {d2[k]}; retraining model from scratch'

def is_local_module(module_name):
    spec = importlib.util.find_spec(module_name)
    return spec is not None \
        and spec.origin is not None \
        and os.path.dirname(__file__) in spec.origin
        
def get_imports(module_file_header):
    # find any other files that are imported by the model file
    module_file = module_file_header
    if 'util' in module_file:
        return []
    
    with open(os.path.join(os.getcwd(),f'{module_file}.py'), 'r') as f:
        module_code = f.read()
    imports = re.findall(r'from (.*?) import', module_code)
    imports += re.findall(r'import (.*?)\n', module_code)
    # select imports that are not standard
    return [i.replace('.','/') for i in imports if is_local_module(i)]    

def get_all_imports(module_names, imports):
    for module_name in module_names:
        cim = get_imports(module_name)
        imports.extend(cim)
        if len(cim) > 0 :
            get_all_imports(cim, imports)
    return imports
    

def capture_local_module(module_name):
    # get all modules
    module_file = inspect.getfile(module_name).replace('.py','').replace(os.getcwd(),'')[1:]
    modules = get_all_imports([module_file], [])
    modules.extend([module_file])
    return modules
        
def time_stat(model):
        
    cpath = os.getcwd()
    keys = capture_local_module(model)
    hashes = {
        k: hashlib.md5(open(os.path.join(cpath,f'{k}.py'),'rb').read()).hexdigest() for k in keys
    }
    times = {
        k: os.path.getmtime(os.path.join(cpath,f'{k}.py')) for k in keys
    }
    last_save_time = max(times.values())
    return [last_save_time, hashes, times]

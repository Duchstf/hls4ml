
class Backend(object):
    def __init__(self, name):
        self.name = name
        self.config_templates = {}
        self.function_templates = {}
        self.tcl_templates = {}

    def get_config_template(self, kind):
        return self.config_templates.get(kind)

    def get_function_template(self, kind):
        return self.function_templates.get(kind)

    def get_tcl_template(self, kind):
        return self.tcl_templates.get(kind)

    def register_templates(self, name, function_template, config_template, tcl_template):
        self.function_templates[name] = function_template
        self.config_templates[name] = config_template
        self.tcl_templates[name] = tcl_template

    def register_source(self, file_name, source, destination_dir='nnet_utils'):
        raise NotImplementedError

backend_map = {}

def register_backend(name, backend_cls):
    if name in backend_map:
        raise Exception('Backend {} already registered'.format(name))
    
    backend_map[name] = backend_cls

def get_backend(name):
    return backend_map[name]()

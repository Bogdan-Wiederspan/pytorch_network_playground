PLOT_REGISTRY = {}

def register_plot(name):
    def wrapper(cls):
        PLOT_REGISTRY[name] = cls
        return cls
    return wrapper

from .tusimple import tuSimple

datasets = {
    'tusimple': tuSimple,
}

def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
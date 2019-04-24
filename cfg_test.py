from configparser import RawConfigParser

cfg = RawConfigParser()
cfg.read('./model.cfg')

config =    {

    'img_H' : int(cfg.get('img config', 'img_H')), 
    'img_W' : int(cfg.get('img config', 'img_W')), 
    'max_true_boxes' : int(cfg.get('img config', 'max_true_boxes')),
    'grid_H' : int(cfg.get('img config', 'grid_H')), 
    'grid_W' : int(cfg.get('img config', 'grid_W')),
    'box' : int(cfg.get('img config', 'box')), 
    'classes': int(cfg.get('img config', 'classes')),
    'anchors' : [float(x) for x in list(cfg.get('img config', 'anchors').split(', '))],
    'batch_size' : int(cfg.get('img config', 'classes')),
    'warm_up_epochs' : int(cfg.get('img config', 'warm_up_epochs'))
}
print(config)
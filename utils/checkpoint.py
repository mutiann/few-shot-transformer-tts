import os
from collections import OrderedDict
import torch
import glob
import logging


def find_ckpt(base_dir):
    max_step = 0
    result = None
    for f in glob.iglob(os.path.join(base_dir, 'model.ckpt-*')):
        step = int(f.split('-')[-1])
        if step > max_step:
            result = f
            max_step = step
    return result


def save_model(model_dir, model=None, optim=None, sched=None, step=None):
    state_dict = {}
    if model:
        if hasattr(model, 'module'):
            model_dict = model.module.state_dict()
        else:
            model_dict = model.state_dict()
        state_dict['model'] = model_dict
    if optim:
        state_dict['optim'] = optim.state_dict()
    if sched:
        state_dict['sched'] = sched.state_dict()
    if step:
        state_dict['step'] = step
        model_dir = os.path.join(model_dir, 'model.ckpt-%d' % step)
    torch.save(state_dict, model_dir)


def load_model(model_path, model=None, optim=None, sched=None, map_location={}):
    state_dict = torch.load(model_path, map_location=map_location)
    if 'model' in state_dict and model:
        model_dict = state_dict['model']
        if hasattr(model, 'module'):
            model.module.load_state_dict(model_dict)
        else:
            model.load_state_dict(model_dict)
    if 'optim' in state_dict and optim:
        optim.load_state_dict(state_dict['optim'])
    if 'step' in state_dict:
        step = state_dict['step']
    else:
        step = None
    if 'sched' in state_dict and sched:
        sched.load_state_dict(state_dict['sched'])
        if step:
            if step != sched.last_epoch:
                logging.warn("Step=%d, while in sched step=%d" % (step, sched.last_epoch))
        else:
            step = sched.last_epoch
    return step

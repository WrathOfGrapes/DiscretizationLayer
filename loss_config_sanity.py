

def check_error_loss(config):
    if not config['type'] == 'error':
        return False

    if 'parameters' not in config.keys():
        return False
    parameters = config['parameters']

    result = True

    if 'alpha' in parameters.keys():
        result = result and 0 < parameters['alpha'] < 1
    if 'gap' in parameters.keys():
        result = result and (-1 < parameters['gap'] < 1) and \
                 (0 < parameters['alpha'] - parameters['gap'] < 1) and \
                 (0 < parameters['alpha'] + parameters['gap'] < 1)
    if 'target' in parameters.keys():
        result = result and 0 < parameters['target'] < 1
    if 'multiplier' in parameters.keys():
        result = result and abs(parameters['multiplier']) > 1e-4
    if 'shift' in parameters.keys():
        result = result and 0 < parameters['shift'] < 1
    if 'loss_type' in parameters.keys():
        result = result and parameters['loss_type'] in ['bce', 'mse', 'mae']

    return result


def check_shift_loss(config):
    if not config['type'] == 'shift':
        return False

    if 'parameters' not in config.keys():
        return False
    parameters = config['parameters']

    result = True

    if 'alpha' in parameters.keys():
        result = result and  0 < parameters['alpha'] < 1
    if 'gap' in parameters.keys():
        result = result and (-1 < parameters['gap'] < 1) and \
                 (0 < parameters['alpha'] - parameters['gap'] < 1) and \
                 (0 < parameters['alpha'] + parameters['gap'] < 1)
    if 'target' in parameters.keys():
        result = result and  0 < parameters['target'] < 1
    if 'loss_type' in parameters.keys():
        result = result and parameters['loss_type'] in ['bce', 'mse', 'mae']

    return result


def check_classic_loss(config):
    loss_type_condition = config['type'] in ['bce', 'mse', 'mae']
    parameters = config['parameters'] == {}
    return loss_type_condition and parameters


def check_loss(config):
    if not isinstance(config, dict):
        return False
    if 'type' not in config.keys():
        return False
    return check_shift_loss(config) or check_error_loss(config) or check_classic_loss(config)
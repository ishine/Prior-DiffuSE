import yaml
import argparse
from trainer import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--trainer', type=str, default='ComplexDDPMTrainer', help='The trainer to execute')
    parser.add_argument('--config', type=str, default='diff.yml', help='Path to the config file')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--doc', type=str, default='diff', help='A string for documentation purpose')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--assets', type=str, default='assets_dpm', help='Path for saving running related data.')
    parser.add_argument('--generate', action='store_true', help='Whether to test the model')
    parser.add_argument('--retrain', action='store_true', help='Whether to test the model')
    args = parser.parse_args()
    args.log = os.path.join(args.assets, 'log', args.doc)
    args.checkpoint = os.path.join(args.assets, 'checkpoint', args.doc)
    args.generated_wav = os.path.join(args.assets, 'wav', args.doc)


    # parse config file
    with open(os.path.join('conf', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # setup logger
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    if not os.path.exists(args.generated_wav):
        os.makedirs(args.generated_wav)
    handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    # add device
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = prepare_device()
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.benchmark = True

    return args, new_config


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Args = {}".format(args))
    logging.info("Config = {}".format(config))

    '''
    load trainer
    '''
    trainer = eval(args.trainer)(args, config)
    if args.generate:
        trainer.generate_wav(load_pre_train=True)
    else:
        # trainer.train()
        trainer.train_ddpm()

if __name__ == '__main__':
    main()

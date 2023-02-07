from src.utils import EarlyStopping
from src.utils import set_seed
from src.models import Trainer

import logging
import torch 
import yaml


def main(params):

    set_seed(params['seed'])

    # dataset & dataloader
    train_ds, val_ds, test_ds = None, None, None

    # check wheter the y is included or not
    logging.info(f'Training set size: {len(train_ds)}')
    logging.info(f'Validation set size: {len(val_ds)}')
    logging.info(f'Test set size: {len(test_ds)}')

    train_loader = None
    val_loader = None
    test_loader = None

    # get model class and trainer
    model = None
    trainer = Trainer(model, params)

    # load checkpoint, if any
    start_epoch = 0
    if params['load_ckpt_filepath']:
        start_epoch = trainer.load_checkpoint(params['load_ckpt_filepath'])
        logging.info(f'Loaded model checkpoint from Epoch {start_epoch - 1}')

    # set early stopping, if required
    if params['early_stopping']:
        early_stopping = EarlyStopping()

    best_loss = 0

    for epoch in range(start_epoch,params['num_epochs']):
        
        train_metrics = trainer.train(epoch, train_loader)
        val_metrics = trainer.validate(epoch, val_loader)

        # tensorboard stuff..
        # ...
        # ...

        # early stopping
        if params['early_stopping']:
            early_stopping(val_metrics['loss'])

        if params['store_ckpt_path'] and best_loss < val_metrics['loss']:
            best_loss = val_metrics['loss']
            full_filepath = trainer.save_checkpoint(epoch, params['store_ckpt_path'], train_metrics, val_metrics)
            logging.info(f'Stored checkpoint at \'{full_filepath}\'')


        print("\nEpoch {}: train_loss: {:.3f} val_loss: {:.3f}".format(str(epoch).zfill(2), 
                                                                       train_metrics['loss'], 
                                                                       val_metrics['loss']))

        # todo: test pipeline on trainer! 
        trainer.test(val_loader)
        logging.info(f'Stored results at \'{params["results_path"]}\' ')

if __name__ == "__main__":
    
    
    with open('settings.yaml', 'r') as fyaml:
        params = yaml.load(fyaml, Loader=yaml.FullLoader)

    if params['verbose']:
        logging.getLogger().setLevel(logging.INFO)


    main(params)
    

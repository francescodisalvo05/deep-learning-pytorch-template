from src.utils import EarlyStopping
from src.utils import set_seed
from src.models import Trainer

import torch 
import yaml


def main(params):

    # dataset & dataloader
    train_loader = None
    val_loader = None

    # get model class and trainer
    model = None
    trainer = Trainer(model, params)

    # load checkpoint, if any
    start_epoch = 0
    if params['load_ckpt_filepath']:
        start_epoch = trainer.load_checkpoint(params['load_ckpt_filepath'])

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
            trainer.save_checkpoint(epoch, params['store_ckpt_path'], train_metrics, val_metrics)

        print("\nEpoch {}: train_loss: {:.3f} val_loss: {:.3f}".format(str(epoch).zfill(2), 
                                                                       train_metrics['loss'], 
                                                                       val_metrics['loss']))


if __name__ == "__main__":
    
    
    with open('settings.yaml', 'r') as fyaml:
        params = yaml.load(fyaml, Loader=yaml.FullLoader)

    set_seed(params['seed'])

    main(params)
    

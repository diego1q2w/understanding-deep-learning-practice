from config import ExperimentConfig
from gradient_utils import get_conv_grad_magnitudes
from factories import (
    create_model, create_optimizer, create_dataloaders,
    create_scheduler, set_seed
)
import torch
import torch.nn as nn
from trainer import Trainer
import pickle

def create_and_run_trainner(config: ExperimentConfig):
    set_seed(config.seed)

    train_loader, val_loader = create_dataloaders(
        config.data,
        config.training.batch_size
    )

    if config.device:
        device = config.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print("device: ", device)
    model = create_model(config.model, device)

    def init_weights(m):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    model.apply(init_weights)

    optimizer = create_optimizer(config.optimizer, model.parameters())

    scheduler = create_scheduler(config.scheduler, optimizer)

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        gradient_fn=get_conv_grad_magnitudes,
        enable_gradient_tracking=config.training.enable_gradient_tracking
    )

    trainer.fit(config.training.epochs)
    results = {
        'config': config.to_dict(),
        'train_losses': trainer.train_losses,
        'train_accs': trainer.train_accs,
        'val_losses': trainer.val_losses,
        'val_accs': trainer.val_accs,
        'epoch_times': trainer.epoch_times,
        'gradient_history': trainer.gradient_history.numpy() if trainer.gradient_history is not None else None,
    }
    variant_name = config.model.net_type  # 'pre', 'post', or 'nobn'
    with open(f'results_{variant_name}.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"✅ Saved results to results_{variant_name}.pkl")
    return results

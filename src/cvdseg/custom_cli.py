from pytorch_lightning.cli import LightningCLI
from torch.optim.lr_scheduler import PolynomialLR
from omegaconf import OmegaConf

class CustomLightningCLI(LightningCLI):

    @staticmethod
    def configure_optimizers(lightning_module, optimizer, lr_scheduler=None):

        if lr_scheduler is None:
            return optimizer
        
        if isinstance(lr_scheduler, PolynomialLR):
            # total_iters = (len(lightning_module.trainer.datamodule.train_dataloader()) / lightning_module.trainer.accumulate_grad_batches) * lightning_module.trainer.max_epochs
            total_iters = lightning_module.trainer.max_epochs
            lr_scheduler.total_iters = total_iters
            print('\n\n\nTotal steps:', lr_scheduler.total_iters, '\n\n\n')
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"},
            }
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"},
        }


# OmegaConf resolvers
def length(lst: list) -> int:
    return len(lst)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("length", length)
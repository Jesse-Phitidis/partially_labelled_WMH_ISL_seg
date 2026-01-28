from cvdseg.lightning_modules.multi_class import Multiclass 
from cvdseg.lightning_modules import utils
import torch

class Stage2(Multiclass): 
    
    def __init__(self, s1_ckpt_path, first_run, **kwargs):
        super().__init__(**kwargs)
        self.s1_ckpt_path = s1_ckpt_path
        self.first_run = first_run
    
    def on_fit_start(self):
        super().on_fit_start()
        if self.first_run:
            state_dict = {utils.split_join(k, ".", slice(1, None)): v for k, v in torch.load(self.s1_ckpt_path)["state_dict"].items() if k.startswith("network")}
            self.network.load_state_dict(state_dict)
            out_channels = 1 + len(self.trainer.datamodule.labels)
            utils.recursive_replace(self.network, utils.out_channels_replace_fn, out_channels=out_channels)
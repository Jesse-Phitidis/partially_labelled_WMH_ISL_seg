import pytorch_lightning as pl
from cvdseg.custom_cli import CustomLightningCLI

def main():
    cli = CustomLightningCLI(
        pl.LightningModule,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__":
    main()
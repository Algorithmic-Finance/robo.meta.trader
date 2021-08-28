import torch as th
import pytorch_lightning as pl

from absl import app, flags, logging

flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_float('lr', 1e-2, '')
flags.DEFINE_float('momentum', 0.9, '')

FLAGS = flags.FLAGS

class ViTClassifier(pl.LightningModule):

    def __init__(self, params):
        super.__init__()
        self.model = None
        self.params = params

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def prepare_data(self):
        print('Creating datasources')
        ds = DataSource()
        ds.add_dataset(params['data']['dataset'], 'train')
        print(f"Total slide count: [{len(ds)}]")
        self.train_ds, self.valid_ds = ds.split(0.10)

    def train_dataloader(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = FLAGS.lr)
        scheduler = scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer)
        return [optimizer], [scheduler]


import torch
import pytorch_lightning as pl


class Base(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
    
    def predict_step(self, batch, batch_idx):
        return self(batch)

    def any_epoch_end(self, outputs, set_prefix: str):
        metric_names = outputs[0].keys()
        metric_names = [metric_name for metric_name in metric_names
                        if metric_name.startswith(set_prefix)]

        for metric_name in metric_names:
            metric_values = [outputs[i][metric_name]
                             for i in range(len(outputs))]
            avg_metric_value = torch.mean(torch.Tensor(metric_values))
            self.log(metric_name, avg_metric_value)

    def any_step(self, batch, batch_idx, set_prefix: str):
        xb, yb = batch
        yb_pred = self(xb)

        loss = self.loss_func(yb_pred, yb)
        outputs = {
            f'{set_prefix}_loss': loss.detach(),
        }

        if set_prefix == 'train':
            outputs['loss'] = loss

        # batch size of one will create problems for metrics
        if yb.dim() == 1:
            yb = torch.unsqueeze(yb, dim=-1)

        # convert categorical matrix to labels
        yb_true = torch.argmax(yb, dim=-1)

        for metric_name, metric_func in self.metrics.items():
            metric_value = metric_func(yb_pred, yb_true).detach()
            outputs[f'{set_prefix}_{metric_name}'] = metric_value
        
        return outputs
    
    def test_epoch_end(self, outputs):
        return self.any_epoch_end(outputs, set_prefix='test')

    def test_step(self, batch, batch_idx):
        return self.any_step(batch, batch_idx, set_prefix='test')

    def training_epoch_end(self, outputs):
        return self.any_epoch_end(outputs, set_prefix='train')

    def training_step(self, batch, batch_idx):
        return self.any_step(batch, batch_idx, set_prefix='train')

    def validation_epoch_end(self, outputs):
        return self.any_epoch_end(outputs, set_prefix='val')

    def validation_step(self, batch, batch_idx):
        return self.any_step(batch, batch_idx, set_prefix='val')

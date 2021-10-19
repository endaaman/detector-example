import os
import io
import json
import yaml
import time
import datetime

from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import matplotlib
from matplotlib import ticker, pyplot as plt
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from effdet import EfficientDet, DetBenchTrain, get_efficientdet_config
from effdet.efficientdet import HeadNet

from datasets import CircleDataset
from models import Darknet, SSD300
from models.ssd import MultiBoxLoss
from endaaman import TorchCommander


SIZE_BY_DEPTH = {
    'd0': 128 * 4,
    'd1': 128 * 5,
    'd2': 128 * 6,
    'd3': 128 * 7,
    'd4': 128 * 8,
    'd5': 128 * 10,
    'd6': 128 * 12,
    'd7': 128 * 14,
}


class MyTrainer(TorchCommander):
    def arg_common(self, parser):
        parser.add_argument('-e', '--epoch', type=int, default=50)
        parser.add_argument('-b', '--batch-size', type=int, default=16)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--workers', type=int, default=os.cpu_count()//2)
        parser.add_argument('--no-aug', action='store_true')
        parser.add_argument('-s', '--suffix', type=str, default='')
        parser.add_argument('--no-skip-first', action='store_true')
        parser.add_argument('-p', '--period-save-weight', type=int, default=10)
        parser.add_argument('-n', '--no-show-fig', action='store_true')

    def train_model(self, model, loaders, eval_fn, additional_metrics={}, save_hook=None):
        assert self.model_name
        assert self.type_name
        matplotlib.use('Agg')
        train_loader = loaders.get('train')
        val_loader = loaders.get('val')
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        print('Starting training')
        train_history = {'loss':[], **{k:[] for k in additional_metrics.keys()}}
        val_history = {'loss':[], **{k:[] for k in additional_metrics.keys()}}
        writer = SummaryWriter(log_dir='log/', filename_suffix=self.type_name + '_' + self.model_name)
        for epoch in range(1, self.args.epoch + 1):
            header = f'[{epoch}/{self.args.epoch}] '

            lr = optimizer.param_groups[0]['lr']
            now = datetime.datetime.now().strftime('%H:%M:%S')
            print(f'{header}starting lr={lr:.7f} ({now})')

            train_metrics = {'loss':[], **{k:[] for k in additional_metrics.keys()}}
            val_metrics = {'loss':[], **{k:[] for k in additional_metrics.keys()}}

            train_start_time = time.perf_counter()
            t = tqdm(train_loader, leave=False)
            for (inputs, labels) in t:
                optimizer.zero_grad()
                loss, outputs = eval_fn(inputs, labels)
                loss.backward()
                optimizer.step()
                train_metrics['loss'].append(float(loss.item()))
                # outputs = outputs.cpu().detach()
                for k, metrics_fn in additional_metrics.items():
                    v = metrics_fn(outputs, labels)
                    train_metrics[k].append(v)
                message = ' '.join([f'{k}:{v[-1]:.4f}' for k, v in train_metrics.items()])
                t.set_description(f'{header}{message}')
                t.refresh()
            for k, v in train_metrics.items():
                train_history[k].append(np.mean(v))
            train_message = ' '.join([f'{k}:{v[-1]:.4f}' for k, v in train_history.items()])
            train_duration = str(datetime.timedelta(seconds=int(time.perf_counter() - train_start_time)))
            print(f'{header}train: {train_message} ({train_duration})')

            #* validate
            if val_loader:
                model.eval()
                with torch.set_grad_enabled(False):
                    for i, (inputs, labels) in enumerate(val_loader):
                        loss, outputs = eval_fn(inputs, labels)
                        # outputs = outputs.cpu().detach()
                        val_metrics['loss'].append(float(loss.item()))
                        for k, metrics_fn in additional_metrics.items():
                            val_metrics[k].append(metrics_fn(outputs, labels))
                model.train()
                for k, v in val_metrics.items():
                    val_history[k].append(np.mean(v))
                val_message = ' '.join([f'{k}:{v[-1]:.4f}' for k, v in val_history.items()])
                print(f'{header}val: {val_message}')

            #* draw fig
            if epoch > 1:
                fisrt_idx = 0 if self.args.no_skip_first else 1
                x_axis = np.arange(1, epoch+1)[fisrt_idx:]
                fig = plt.figure(figsize=(10, 5))
                for i, (k, train_values) in enumerate(train_history.items()):
                    ax = fig.add_subplot(1, len(train_history.keys()), i+1)
                    ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
                    ax.set_title(k)
                    ax.plot(x_axis, train_values[fisrt_idx:], label=f'train')
                    if val_loader:
                        val_values = val_history[k]
                        ax.plot(x_axis, val_values[fisrt_idx:], label=f'val')
                    ax.legend()
                fig_path = f'out/training_curve_{full_name}.png'
                plt.savefig(fig_path)

                if epoch == 2 and not self.args.no_show_fig:
                    os.system(f'xdg-open {fig_path} > /dev/null')
                fig.clf()
                plt.clf()
                plt.close()

            for (tag, metrics) in (('train', train_metrics), ('val', val_metrics)):
                for k, v in metrics.items():
                    writer.add_scalar(f'{k}/{tag}', v, e-1)

            #* save weights
            if epoch % self.args.period_save_weight == 0:
                weights_path = self.save_weights(model, epoch, train_history, val_history, save_hook)
                print(f'{header}Saved "{weights_path}"')

            scheduler.step(train_metrics['loss'][-1])
            # scheduler.step()
            print()

    def save_weights(self, model, epoch, train_history, val_history, save_hook=None):
        weights_dir = f'weights/{self.type_name}/{self.model_name}'
        if self.args.suffix:
            weights_dir += '_' + self.args.suffix
        os.makedirs(weights_dir, exist_ok=True)

        weights_name = f'{epoch}.pth'
        weights = {
            'type_name': self.type_name,
            'model_name': self.model_name,
            'suffix': self.args.suffix,
            'epoch': epoch,
            'args': self.args,
            'state_dict': value,
            'train_history': train_history,
            'val_history': val_history,
        }

        if callable(save_hook):
            weights_dir, weights_name, weights = save_hook(weights_dir, weights_name, weights)

        weights_path = os.path.join(weights_dir, weights_name)
        torch.save(state, weights_path)
        return weights_path

    def create_loaders(self, target, image_size):
        if target not in ['effdet', 'yolo', 'ssd']:
            raise ValueError(f'Invalid target: {target}')
        train_dataset = CircleDataset(target=target)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )
        return {
            'train': train_loader,
        }


    def arg_effdet(self, parser):
        parser.add_argument('-d', '--depth', default='d0', choices=list(SIZE_BY_DEPTH.keys()))

    def pre_effdet(self):
        self.type_name = 'effdet'
        self.model_name = self.args.depth

    def run_effdet(self):
        cfg = get_efficientdet_config(f'tf_efficientdet_{self.args.depth}')
        cfg.num_classes = 6
        model = EfficientDet(cfg)
        bench = DetBenchTrain(model).to(self.device)
        loaders = self.create_loaders('effdet', SIZE_BY_DEPTH[self.args.depth])

        def eval_fn(inputs, labels):
            inputs = inputs.to(self.device)
            labels['bbox'] = labels['bbox'].to(self.device)
            labels['cls'] = labels['cls'].to(self.device)
            loss = bench(inputs, labels)
            return loss['loss'], None

        self.train_model(
            model,
            loaders,
            eval_fn, {
                # metrics_fn
            })

    def pre_yolo(self):
        self.type_name = self.model_name = 'yolo'

    def run_yolo(self):
        model = Darknet().to(self.device)
        model.train()
        loaders = self.create_loaders('yolo', 416)

        def eval_fn(inputs, labels):
            for idx, ll in enumerate(labels):
                ll[:, 0] = idx
            inputs = inputs.to(self.device)
            labels = labels.view(-1, 6).to(self.device) # batch x [batch_idx, cls_id, x, y, w, h]
            loss, outputs = model(inputs, labels)
            return loss, outputs

        def save_hook(weights_dir, weights_name, weights):
            name = weights['epoch'] + '.darknet'
            model.save_darknet_weights(os.path.join(weights_dir, name))
            weights['darknet_weight'] = name
            return weights_dir, weights_name, weights

        self.train_model(
            model,
            loaders,
            eval_fn, {
                # metrics_fn
            })

    def pre_ssd(self):
        self.type_name = self.model_name = 'ssd'

    def run_ssd(self):
        model = SSD300(n_classes=7).to(self.device)
        loaders = self.create_loaders('ssd', 300)
        criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy)

        def eval_fn(inputs, labels):
            inputs = inputs.to(self.device)
            bboxes = [b.to(self.device) for b in labels[0]]
            clss = [c.to(self.device) for c in labels[1]]
            predicted_locs, predicted_scores = model(inputs)
            loss = criterion(predicted_locs, predicted_scores, bboxes, clss)
            return loss, None

        self.train_model(
            model,
            loaders,
            eval_fn, {
                # metrics_fn
            })

    def run_fake_weights(self):
        model = self.create_model(self.args.network)
        weights_path = self.save_weights(model, 0)
        print(weights_path)

MyTrainer().run()

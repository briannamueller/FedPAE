"""
Note: minimal opt-in client-side train/val split (use_val/val_ratio/split_seed).
Default behavior remains unchanged when use_val is False.
"""
import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from flcore.trainmodel.models import BaseHeadSplit
try:
    from torchmetrics.functional import multiclass_recall  # type: ignore
except ImportError:  # torchmetrics < 0.11
    from torchmetrics.functional.classification import multiclass_recall  # type: ignore

class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.args = args
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.role = 'Client_' + str(self.id)
        self.save_folder_name = args.save_folder_name_full

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot
        self.use_val = args.use_val
        self.val_ratio = args.val_ratio
        self.split_seed = args.split_seed
        self.use_bacc_metric = bool(getattr(args, "use_bacc_metric", False))

        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            if args.models_folder_name:
                model = load_item(self.role, 'model', args.models_folder_name).to(self.device)
                print('load pre-trained model from', args.models_folder_name)
            else:
                model = BaseHeadSplit(args, self.id).to(self.device)
            save_item(model, self.role, 'model', self.save_folder_name)

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self._best_val_acc = -1.0
        self._best_val_bacc = -1.0

        if getattr(args, "weighted_loss", False):
            train_ds = read_client_data(args.dataset, self.id, is_train=True, few_shot=self.few_shot)
            counts = torch.zeros(args.num_classes, dtype=torch.float)
            for _, y in train_ds:
                lbl = int(y.item()) if torch.is_tensor(y) else int(y)
                if 0 <= lbl < counts.numel():
                    counts[lbl] += 1.0
            safe = counts.clamp(min=1.0)
            weights = (safe.sum() / safe.numel()) / safe
            self.loss = nn.CrossEntropyLoss(weight=weights.to(self.device))
        else:
            self.loss = nn.CrossEntropyLoss()


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        if self.use_val:
            full_dataset = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
            g = torch.Generator().manual_seed(self.split_seed)
            train_size = int((1.0 - self.val_ratio) * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_data, _ = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=g)
            return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_val_data(self, batch_size=None):
        if not self.use_val:
            raise RuntimeError("Validation requested but use_val is False.")
        if batch_size == None:
            batch_size = self.batch_size
        full_dataset = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        g = torch.Generator().manual_seed(self.split_seed)
        train_size = int((1.0 - self.val_ratio) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        _, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=g)
        return DataLoader(val_data, batch_size, drop_last=False, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

    def balanced_accuracy(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """Macro recall over classes with support, matching sklearn balanced_accuracy_score."""
        preds = preds.long()
        targets = targets.long()
        recalls = multiclass_recall(
            preds,
            targets,
            num_classes=self.num_classes,
            average=None,
        )
        support = torch.bincount(targets, minlength=self.num_classes).to(recalls.device) > 0
        if support.any():
            return float(recalls[support].mean().item())
        return float('nan')
    
    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    # def test_metrics(self):
    #     testloader = self.load_test_data()
    #     model = load_item(self.role, 'model', self.save_folder_name)
    #     # model.to(self.device)
    #     model.eval()

    #     test_acc = 0
    #     test_num = 0
    #     y_prob = []
    #     y_true = []
        
    #     with torch.no_grad():
    #         for x, y in testloader:
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             output = model(x)

    #             test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
    #             test_num += y.shape[0]

    #             y_prob.append(output.detach().cpu().numpy())
    #             nc = self.num_classes
    #             if self.num_classes == 2:
    #                 nc += 1
    #             lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
    #             if self.num_classes == 2:
    #                 lb = lb[:, :2]
    #             y_true.append(lb)

    #     y_prob = np.concatenate(y_prob, axis=0)
    #     y_true = np.concatenate(y_true, axis=0)

    #     auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
    #     return test_acc, test_num, auc

    def _evaluate_loader(self, loader):
        """Shared evaluation logic. Returns (num_correct, num_samples, auc, bacc)."""
        model = load_item(self.role, 'model', self.save_folder_name)
        model.to(self.device)
        model.eval()

        num_correct = 0
        num_samples = 0
        preds_all = []
        targets_all = []
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                preds = torch.argmax(output, dim=1)
                num_correct += (torch.sum(preds == y)).item()
                num_samples += y.shape[0]
                preds_all.append(preds.detach().cpu().numpy())
                targets_all.append(y.detach().cpu().numpy())

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes + (1 if self.num_classes == 2 else 0)
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        try:
            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        except ValueError:
            auc = float("nan")

        if len(preds_all) == 0:
            return 0, 0, float("nan"), float("nan"), None, None, None
        preds_cat = np.concatenate(preds_all, axis=0)
        targets_cat = np.concatenate(targets_all, axis=0)
        bacc = metrics.balanced_accuracy_score(targets_cat, preds_cat)

        return num_correct, num_samples, auc, bacc, preds_cat, y_prob, targets_cat

    def test_metrics(self):
        """Returns (num_correct, num_samples, auc, bacc)."""
        testloader = self.load_test_data()
        result = self._evaluate_loader(testloader)
        return result[0], result[1], result[2], result[3]

    def val_metrics(self):
        """Returns (num_correct, num_samples, auc, bacc)."""
        if not self.use_val:
            raise RuntimeError("Validation requested but use_val is False.")
        valloader = self.load_val_data()
        result = self._evaluate_loader(valloader)
        # Save best checkpoint by val acc and val bacc
        num_correct, num_samples, auc, bacc = result[0], result[1], result[2], result[3]
        if num_samples > 0:
            acc = num_correct / num_samples
            if acc > self._best_val_acc:
                self._best_val_acc = acc
                model = load_item(self.role, 'model', self.save_folder_name)
                save_item(model, self.role, 'best_acc_model', self.save_folder_name)
            if bacc > self._best_val_bacc:
                self._best_val_bacc = bacc
                model = load_item(self.role, 'model', self.save_folder_name)
                save_item(model, self.role, 'best_bacc_model', self.save_folder_name)
        return num_correct, num_samples, auc, bacc

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        # model.to(self.device)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


def save_item(item, role, item_name, item_path=None):
    if not os.path.exists(item_path):
        os.makedirs(item_path)
    torch.save(item, os.path.join(item_path, role + "_" + item_name + ".pt"))

def load_item(role, item_name, item_path=None):
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        use_cuda = torch.cuda.is_available() and visible not in ("", "-1")
        if use_cuda:
            map_location = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            map_location = "mps"
        else:
            map_location = "cpu"
        item = torch.load(
            os.path.join(item_path, role + "_" + item_name + ".pt"),
            map_location=map_location,
            weights_only=False,
        )
        if isinstance(item, torch.nn.Module):
            try:
                item = item.to(map_location)
            except Exception:
                pass
        return item
    except FileNotFoundError:
        print(role, item_name, 'Not Found')
        return None

import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import shutil
import json
from pathlib import Path
from utils.data_utils import read_client_data
from flcore.clients.clientbase import load_item, save_item


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break
        self.role = 'Server'
        pfl_fp = getattr(args, "pfl_config_id", "")
        if args.save_folder_name == 'temp':
            args.save_folder_name_full = f'{args.save_folder_name}/{args.algorithm}/{time.time()}/'
        elif 'temp' in args.save_folder_name:
            args.save_folder_name_full = args.save_folder_name
        elif pfl_fp:
            args.save_folder_name_full = f'{args.save_folder_name}/{args.algorithm}/pfl[{pfl_fp}]/'
        else:
            args.save_folder_name_full = f'{args.save_folder_name}/{args.algorithm}/'
        self.save_folder_name = args.save_folder_name_full
        self.resume_from = os.fspath(Path(getattr(args, "resume_from", "")).expanduser()) if getattr(args, "resume_from", "") else ""
        self.state_dir = Path(self.save_folder_name)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.start_round = 0
        if self.resume_from:
            self._copy_resume_artifacts()
            state = self.load_state()
            if state and "last_round" in state:
                self.start_round = int(state["last_round"]) + 1

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_test_bacc = []
        self.rs_train_loss = []
        self.rs_val_acc = []
        self.rs_val_auc = []
        self.rs_val_bacc = []
        # Per-client metrics: list of lists, one inner list per eval round
        self.pc_test_acc = []
        self.pc_test_auc = []
        self.pc_test_bacc = []
        self.pc_val_acc = []
        self.pc_val_auc = []
        self.pc_val_bacc = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate
        self.use_val = bool(getattr(args, "use_val", False))
        self.use_bacc_metric = bool(getattr(args, "use_bacc_metric", False))

    def _copy_resume_artifacts(self):
        if not self.resume_from or not os.path.isdir(self.resume_from):
            return
        if os.path.abspath(self.resume_from) == os.path.abspath(self.save_folder_name):
            return
        os.makedirs(self.save_folder_name, exist_ok=True)
        for fname in os.listdir(self.resume_from):
            src = os.path.join(self.resume_from, fname)
            if not os.path.isfile(src):
                continue
            if not fname.endswith(".pt"):
                continue
            dst = os.path.join(self.save_folder_name, fname)
            shutil.copy2(src, dst)

    def _state_path(self):
        return self.state_dir / "state.json"

    def save_state(self, round_idx):
        payload = {
            "last_round": int(round_idx),
            "global_rounds": int(self.global_rounds),
            "dataset_tag": getattr(self.args, "dataset_tag", self.dataset),
            "algorithm": self.algorithm,
            "save_folder_name": self.save_folder_name,
            "timestamp": time.time(),
        }
        path = self._state_path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(path)

    def load_state(self):
        path = self._state_path()
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_parameters(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_ids(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        client = self.clients[self.uploaded_ids[0]]
        global_model = load_item(client.role, 'model', client.save_folder_name)
        for param in global_model.parameters():
            param.data.zero_()
            
        for w, cid in zip(self.uploaded_weights, self.uploaded_ids):
            client = self.clients[cid]
            client_model = load_item(client.role, 'model', client.save_folder_name)
            for server_param, client_param in zip(global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * w

        save_item(global_model, self.role, 'global_model', self.save_folder_name)
        
    def save_results(self):
        pfl_fp = getattr(self.args, "pfl_config_id", "")
        # Dataset name is already encoded in the results/ directory path (nested layout),
        # so the h5 filename only needs algo + fingerprint.
        algo = self.algorithm
        if pfl_fp:
            algo += f"_pfl[{pfl_fp}]"
        result_path = getattr(self.args, "results_dir", "../results")
        result_path = os.fspath(result_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_name = "{}.h5".format(algo)
            if self.start_round > 0 or getattr(self.args, "resume_from", ""):
                file_name = "{}_resume_round{}.h5".format(algo, self.start_round)
            file_path = os.path.join(result_path, file_name)
            file_dir = os.path.dirname(file_path)
            if file_dir and not os.path.exists(file_dir):
                os.makedirs(file_dir)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                # ---- Global aggregates (1-D: num_rounds) ----
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_test_bacc', data=self.rs_test_bacc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

                if len(self.rs_val_acc):
                    hf.create_dataset('rs_val_acc', data=self.rs_val_acc)
                    hf.create_dataset('rs_val_auc', data=self.rs_val_auc)
                    hf.create_dataset('rs_val_bacc', data=self.rs_val_bacc)

                # ---- Per-client metrics (2-D: num_rounds × num_clients) ----
                if len(self.pc_test_acc):
                    hf.create_dataset('per_client_test_acc', data=np.array(self.pc_test_acc))
                    hf.create_dataset('per_client_test_auc', data=np.array(self.pc_test_auc))
                    hf.create_dataset('per_client_test_bacc', data=np.array(self.pc_test_bacc))

                if len(self.pc_val_acc):
                    hf.create_dataset('per_client_val_acc', data=np.array(self.pc_val_acc))
                    hf.create_dataset('per_client_val_auc', data=np.array(self.pc_val_auc))
                    hf.create_dataset('per_client_val_bacc', data=np.array(self.pc_val_bacc))
        
        if 'temp' in self.save_folder_name:
            try:
                shutil.rmtree(self.save_folder_name)
                print('Deleted.')
            except:
                print('Already deleted.')

    # def test_metrics(self):
    #     num_samples = []
    #     tot_correct = []
    #     tot_auc = []
    #     for c in self.clients:
    #         ct, ns, auc = c.test_metrics()
    #         tot_correct.append(ct*1.0)
    #         print(f'Client {c.id}: Acc: {ct*1.0/ns}, AUC: {auc}')
    #         tot_auc.append(auc*ns)
    #         num_samples.append(ns)

    #     ids = [c.id for c in self.clients]

    #     return ids, num_samples, tot_correct, tot_auc

    def test_metrics(self):
        """Returns (ids, num_samples, tot_correct, tot_auc, tot_bacc)."""
        num_samples = []
        tot_correct = []
        tot_auc = []
        tot_bacc = []
        for c in self.clients:
            ct, ns, auc, bacc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            tot_bacc.append(bacc)
            num_samples.append(ns)
            acc = ct * 1.0 / ns if ns > 0 else float('nan')
            print(f'Client {c.id}: Acc: {acc:.4f}, BAcc: {bacc:.4f}, AUC: {auc:.4f}')
        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct, tot_auc, tot_bacc

    def val_metrics(self):
        """Returns (ids, num_samples, tot_correct, tot_auc, tot_bacc)."""
        if not getattr(self, "use_val", False):
            raise RuntimeError("Validation requested but use_val is False.")
        num_samples = []
        tot_correct = []
        tot_auc = []
        tot_bacc = []
        for c in self.clients:
            ct, ns, auc, bacc = c.val_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            tot_bacc.append(bacc)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct, tot_auc, tot_bacc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
            print(f'Client {c.id}: Loss: {cl*1.0/ns}')

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def evaluate(self, acc=None, loss=None):
        ids, num_samples, tot_correct, tot_auc, tot_bacc = self.test_metrics()

        # Per-client metrics
        accs = [c / n if n > 0 else float('nan') for c, n in zip(tot_correct, num_samples)]
        aucs = [a / n if n > 0 else float('nan') for a, n in zip(tot_auc, num_samples)]
        baccs = tot_bacc  # already per-client

        # Global aggregates
        test_acc = sum(tot_correct) / sum(num_samples)
        test_auc = sum(tot_auc) / sum(num_samples)
        test_bacc = float(np.nanmean(baccs))

        # Persist
        if acc is None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        self.rs_test_auc.append(test_auc)
        self.rs_test_bacc.append(test_bacc)
        self.pc_test_acc.append(accs)
        self.pc_test_auc.append(aucs)
        self.pc_test_bacc.append(baccs)

        print(f"Averaged Test Accuracy: {test_acc:.4f}")
        print(f"Averaged Test BAcc: {test_bacc:.4f}")
        print(f"Averaged Test AUC: {test_auc:.4f}")
        print(f"Std Test Accuracy: {np.std(accs):.4f}")
        print(f"Std Test BAcc: {np.std(baccs):.4f}")
        print(f"Std Test AUC: {np.std(aucs):.4f}")

        # Optional val metrics
        if self.use_val:
            _, val_ns, val_correct, val_auc_tot, val_bacc_tot = self.val_metrics()
            val_accs = [c / n if n > 0 else float('nan') for c, n in zip(val_correct, val_ns)]
            val_aucs = [a / n if n > 0 else float('nan') for a, n in zip(val_auc_tot, val_ns)]
            val_baccs = val_bacc_tot

            val_acc = sum(val_correct) / sum(val_ns)
            val_auc = sum(val_auc_tot) / sum(val_ns)
            val_bacc = float(np.nanmean(val_baccs))

            self.rs_val_acc.append(val_acc)
            self.rs_val_auc.append(val_auc)
            self.rs_val_bacc.append(val_bacc)
            self.pc_val_acc.append(val_accs)
            self.pc_val_auc.append(val_aucs)
            self.pc_val_bacc.append(val_baccs)

            print(f"Averaged Val Accuracy: {val_acc:.4f}")
            print(f"Averaged Val BAcc: {val_bacc:.4f}")
            print(f"Averaged Val AUC: {val_auc:.4f}")
            print(f"Std Val Accuracy: {np.std(val_accs):.4f}")
            print(f"Std Val BAcc: {np.std(val_baccs):.4f}")
            print(f"Std Val AUC: {np.std(val_aucs):.4f}")


    # # evaluate selected clients
    # def evaluate(self, acc=None, loss=None):
    #     stats = self.test_metrics()
    #     # stats_train = self.train_metrics()

    #     test_acc = sum(stats[2])*1.0 / sum(stats[1])
    #     test_auc = sum(stats[3])*1.0 / sum(stats[1])
    #     # train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
    #     accs = [a / n for a, n in zip(stats[2], stats[1])]
    #     aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
    #     if acc == None:
    #         self.rs_test_acc.append(test_acc)
    #     else:
    #         acc.append(test_acc)
        
    #     # if loss == None:
    #     #     self.rs_train_loss.append(train_loss)
    #     # else:
    #     #     loss.append(train_loss)

    #     # print("Averaged Train Loss: {:.4f}".format(train_loss))
    #     print("Averaged Test Accuracy: {:.4f}".format(test_acc))
    #     print("Averaged Test AUC: {:.4f}".format(test_auc))
    #     # self.print_(test_acc, train_acc, train_loss)
    #     print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
    #     print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

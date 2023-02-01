import torch
import torch.nn.functional as F
from tqdm.auto import tqdm, trange
import torch.optim as optim
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from data import MAGDataset, MAGDatasetMulti
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import Linear
from torch_geometric.seed import seed_everything
from seq_hgnn.model_pyg_rel_pe import SeqHGNNConv, PositionalEncoding
from torchmetrics.functional import f1_score
import wandb
import warnings
import logging
import argparse


class SeqHGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, agg=['sum'], dropout=0.5):
        super().__init__()
        
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.lin_dict = torch.nn.ModuleDict()
        # for node_type in data.node_types:
        #     self.lin_dict[node_type] = Linear(-1, hidden_channels)
        
        self.rel_encoding = torch.nn.ParameterDict()
        for relation_type in data.edge_types:
            self.rel_encoding['__'.join(relation_type)] = None


        if len(agg) < num_layers:
            for _ in range(num_layers - len(agg)):
                agg.append(agg[-1])
        elif len(agg) > num_layers:
            agg = agg[:num_layers]
        self.agg = agg
        
        self.convs = torch.nn.ModuleList()
        for _, l_agg in zip(range(num_layers), self.agg):
            conv = SeqHGNNConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group=l_agg, dropout=dropout, 
                           cross_att=False, num_cross_layer=1)
            self.convs.append(conv)
        self.dropout = dropout
        self.out_channels = out_channels

        self.flatten = False
        self.final_encoder = torch.nn.TransformerDecoderLayer(d_model=self.hidden_channels, nhead=8)
        # decoder_layer = torch.nn.TransformerDecoderLayer(d_model=self.hidden_channels, nhead=8)
        # self.final_encoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=2)
        # self.pe = PositionalEncoding(self.hidden_channels, dropout=0, max_len=32)
        self.lin = torch.nn.Linear(hidden_channels, self.out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        gain = torch.nn.init.calculate_gain('relu')
        for rel in self.rel_encoding.keys():
            v = torch.FloatTensor(self.num_heads, self.hidden_channels // self.num_heads)
            torch.nn.init.xavier_uniform_(v, gain=gain)
            self.rel_encoding[rel] = torch.nn.Parameter(v.reshape(-1))

    def dropout_channel(self, feats):
        if self.training:
            num_samples = int(feats.shape[1] * self.dropout)# self.dropout
            selected_idx = torch.randperm(feats.shape[1], dtype=torch.int64).to(feats.device)[:num_samples]
            feats[:,selected_idx,:] = 0
        return feats

    def forward(self, x_dict, edge_index_dict):
        
        x_input = {}
        device = x_dict[targe_node_type].device
        for t, x_t in x_dict.items():
            if len(x_t.shape) == 2:
                x_t = x_t.unsqueeze(1)
            x_t_inp = []
            for idx, x_t_i in enumerate(torch.split(x_t, 1, dim=1)):
                key = '{}_{}'.format(t, idx)
                if key not in self.lin_dict.keys():
                    self.lin_dict[key] = Linear(-1, self.hidden_channels).to(device)
                x_t_inp.append(self.lin_dict[key](x_t_i))
            x_t_inp = torch.cat(x_t_inp, dim=1)
            x_input[t] = x_t_inp
            
        for conv in self.convs:
            x_input = conv(x_input, edge_index_dict, self.rel_encoding)
        
        out = x_input[targe_node_type]
        # out = self.pe(out)
        q_out = out[:,0,:].unsqueeze(0)
        out = self.dropout_channel(out)
        
        out = out.transpose(0, 1)
        out= self.final_encoder(q_out, out).squeeze()
        return self.lin(out)


def train():
    model.train()

    total_loss = current_lr = 0
    tr_total_correct = tr_total_examples = 0
    va_total_correct = va_total_examples = 0
    ts_total_correct = ts_total_examples = 0
    batch_count = 0
    for batch_idx in trange(args.n_batch):
        batch = next(iter(train_loader))
        batch = batch.to(device, 'edge_index')
        batch_size = batch[targe_node_type].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = F.cross_entropy(out[:batch_size], batch[targe_node_type][label_name][:batch_size].long())
        pred = out.argmax(dim=-1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        lr_scheduler.step(lr_scheduler.last_epoch+1)

        val_mask = batch[targe_node_type].val_mask.detach()
        test_mask = batch[targe_node_type].test_mask.detach()

        tr_total_examples += batch_size
        va_total_examples += torch.sum(val_mask.int())
        ts_total_examples += torch.sum(test_mask.int())
        total_loss += float(loss) * batch_size
        tr_total_correct += int((pred[:batch_size] == batch[targe_node_type][label_name][:batch_size]).sum())
        va_total_correct += int((pred[val_mask] == batch[targe_node_type][label_name][val_mask]).sum())
        ts_total_correct += int((pred[test_mask] == batch[targe_node_type][label_name][test_mask]).sum())
        current_lr += lr_scheduler.get_lr()[0]*batch_size

    return total_loss / tr_total_examples, tr_total_correct / tr_total_examples, \
        va_total_correct / va_total_examples, ts_total_correct / ts_total_examples, \
            current_lr / tr_total_examples


@torch.no_grad()
def test(loader):
    model.eval()
    total_examples = total_correct = 0
    pred_list = []
    true_list = []
    for batch in tqdm(loader):
        batch = batch.to(device, 'edge_index')
        batch_size = batch[targe_node_type].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)[:batch_size]
        pred = out.argmax(dim=-1)

        total_examples += batch_size
        total_correct += int((pred == batch[targe_node_type][label_name][:batch_size]).sum())
        
        pred_list.append(pred)
        true_list.append(batch[targe_node_type][label_name][:batch_size])
    pred_list = torch.cat(pred_list, dim=-1)
    true_list = torch.cat(true_list, dim=-1)
    return total_correct / total_examples, f1_score(pred_list, true_list, average='macro', num_classes=num_classes)


@torch.no_grad()
def init_params():
    # Initialize lazy parameters via forwarding a single batch to the model:
    batch = next(iter(train_loader))
    batch = batch.to(device, 'edge_index')
    model(batch.x_dict, batch.edge_index_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SeqHGNN')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', choices=['ogbn-mag', 'ogbn-mag-own', 'ogbn-mag-multi', 'ogbn-mag-complex', 'oag-field', 'oag-venue', 'oag-venue-multi', 'recruit', 'hgb-acm','hgb-dblp', 'hgb-imdb'], default='ogbn-mag')
    parser.add_argument('--cls-level', choices=['1', '2', '3', ''], default='')
    parser.add_argument('--num-hidden', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--n-batch', type=int)
    parser.add_argument('--warm-up', type=float, default=0.1)
    parser.add_argument('--num-samples', type=int)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument('--logsubfix', type=str, default='')
    parser.add_argument('--clip', type=float, default=1.0,
                    help='Gradient Norm Clipping')
    parser.add_argument('--agg', nargs='+', type=str, required=False)
    parser.add_argument('--workers', type=int, default=0) 
    parser.add_argument('--save-path')
    parser.add_argument('--off-wandb', default=False, action='store_true')
    args = parser.parse_args()
    tags = [t for t in args.logsubfix.split(',')]
    wandb.init(project='SeqHGNN', entity='Anonymous', tags=tags, mode="disabled" if args.off_wandb else None)
    wandb.config.update(args)
    config = wandb.config
    seed_everything(args.seed)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler("seq_hgnn_pyg_{}_lr{}_wu{}_e{}_drop{}.log".format(
                                args.logsubfix, args.lr, args.warm_up, args.epochs, args.dropout)),
                            logging.StreamHandler()
                        ],
                        level=logging.INFO)
    logging.info(args)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    transform = T.ToUndirected(merge=True)
    if args.dataset == 'ogbn-mag':
        dataset = OGB_MAG('/home/Anonymous/pyg_data/MAG', preprocess='metapath2vec', transform=transform)
        targe_node_type = 'paper'
        num_classes = dataset.num_classes
        threshold = 0.5
        label_name = 'y'
    elif args.dataset == 'ogbn-mag-own':
        dataset = MAGDataset('dataset/OGB_MAG/')
        targe_node_type = 'paper'
        num_classes = dataset.num_classes
        threshold = 0.5
        label_name = 'y'
    elif args.dataset == 'ogbn-mag-multi':
        dataset = MAGDatasetMulti('dataset/OGB_MAG/')
        targe_node_type = 'paper'
        num_classes = dataset.num_classes
        threshold = 0.5
        label_name = 'y'
    data = dataset[0]
    data[targe_node_type].update({'ids': torch.arange(data[targe_node_type].x.shape[0])})
    data = data.to(device, 'x', label_name, 'ids')
    train_input_nodes = (targe_node_type, data[targe_node_type].train_mask)
    val_input_nodes = (targe_node_type, data[targe_node_type].val_mask)
    test_input_nodes = (targe_node_type, data[targe_node_type].test_mask)


    if args.num_samples is None:
        args.num_samples = 6*args.batch_size
    num_samples = []
    train_loader = HGTLoader(data, num_samples=[args.num_samples] * (args.num_layers+1), \
        shuffle=True, input_nodes=train_input_nodes, batch_size=args.batch_size, \
            num_workers=args.workers)
    val_loader = HGTLoader(data, num_samples=[args.num_samples] * (args.num_layers+1),
                        input_nodes=val_input_nodes, batch_size=args.batch_size, 
                           num_workers=args.workers)
    test_loader = HGTLoader(data, num_samples=[args.num_samples] * (args.num_layers+1),
                        input_nodes=test_input_nodes, batch_size=args.batch_size,
                            num_workers=args.workers)
    model = SeqHGNN(hidden_channels=args.num_hidden, out_channels=num_classes, num_heads=args.num_heads, num_layers=args.num_layers, agg=args.agg, dropout=args.dropout).to(device)
    
    init_params()  # Initialize parameters.
    wandb.watch(model)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],     'weight_decay': 0.0}
    ]

    if args.n_batch is None or args.n_batch > len(train_loader):
        args.n_batch = len(train_loader)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-06)
    total_steps = args.n_batch * args.epochs
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.05, anneal_strategy='linear', final_div_factor=10,\
                        max_lr=args.lr, total_steps=total_steps)
    warnings.filterwarnings('ignore')

    best_val_acc = 0
    best_test_acc = 0
    best_val_f1 = 0
    best_test_f1 = 0
    best_epoch = 0

    best_s_val_acc = 0
    best_s_test_acc = 0
    for epoch in range(args.epochs):
        loss, train_acc, s_val_acc, s_test_acc, current_lr = train()
        wandb.log({"Loss/train": loss, 'LR': current_lr, 'Acc/train': train_acc}, step=epoch+1)
        logging.info(f'LR: {current_lr:.4f}, Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, sVal: {s_val_acc:.4f}, sTest: {s_test_acc:.4f}')
        if train_acc >= threshold and (s_val_acc > best_s_val_acc or epoch == args.epochs - 1): # epoch % args.eval_every == 0
            # best_s_val_acc = s_val_acc
            val_acc, val_f1 = test(val_loader)
            test_acc, test_f1 = test(test_loader)
            
            if val_acc > best_val_acc:
                best_s_val_acc = s_val_acc
                best_val_acc = val_acc
                best_epoch = epoch
                best_test_acc = test_acc
                best_val_f1 = val_f1
                best_test_f1 = test_f1
            logging.info(f'Val: {val_acc:.4f}, Test: {test_acc:.4f} Val_F1: {val_f1:.4f}, Test_F1: {test_f1:.4f} BEST_Epoch: {best_epoch:03d}, BEST_Val: {best_val_acc:.4f}, BEST_Test: {best_test_acc:.4f} BEST_Val_F1: {best_val_f1:.4f}, BEST_Test_F1: {best_test_f1:.4f}')

            wandb.log({
                "Acc/validate": val_acc,
                "Acc/test": test_acc,
                "Best/epoch": best_epoch,
                "Best/val_acc": best_val_acc,
                "Best/test_acc": best_test_acc,
                "Best/val_f1": best_val_f1,
                "Best/test_f1": best_test_f1,
            }, step=epoch+1)
        
    logging.info(f'[BEST] Epoch: {best_epoch:03d}, Val: {best_val_acc:.4f}, Test: {best_test_acc:.4f}')
    logging.info(f'[BEST] Val_F1: {best_val_f1:.4f}, Test_F1: {best_test_f1:.4f}')
    wandb.log({
        "Best_Acc/validate": best_val_acc,
        "Best_Acc/test": best_test_acc,
        "Best_F1/validate": best_val_f1,
        "Best_F1/test": best_test_f1,
        "Best_Epoch": best_epoch
    })
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm, trange
import torch.optim as optim
import torch_geometric.transforms as T
from data import load_data
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import Linear
from torch_geometric.seed import seed_everything
from torch_geometric.nn.inits import reset
from seq_hgnn.model import SeqHGNN
from torchmetrics.functional import f1_score, accuracy
import wandb
import warnings
import logging
import argparse


def train():
    model.train()

    loss_list = []
    tr_pred_list, tr_true_list = [], []
    va_pred_list, va_true_list = [], []
    ts_pred_list, ts_true_list = [], []
    data_iter = iter(train_loader)
    for batch_idx in trange(args.n_batch, dynamic_ncols=True):
        batch = next(data_iter)
        batch = batch.to(device, 'edge_index')
        batch_size = batch[targe_node_type].batch_size
        
        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                logist = model(batch.x_dict, batch.edge_index_dict)
                loss = F.cross_entropy(logist[:batch_size], batch[targe_node_type].y[:batch_size])
            scalar.scale(loss).backward()
            scalar.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scalar.step(optimizer)
            scalar.update()
            lr_scheduler.step()
        else:
            logist = model(batch.x_dict, batch.edge_index_dict)
            loss = F.cross_entropy(logist[:batch_size], batch[targe_node_type].y[:batch_size])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            lr_scheduler.step()
        
        pred = logist.argmax(dim=-1).detach().int()
        val_mask = batch[targe_node_type].val_mask
        ts_mask = batch[targe_node_type].test_mask
        
        tr_pred_list.append(pred[:batch_size])
        va_pred_list.append(pred[val_mask])
        ts_pred_list.append(pred[ts_mask])
        
        true = batch[targe_node_type].y.int()
        tr_true_list.append(true[:batch_size])
        va_true_list.append(true[val_mask])
        ts_true_list.append(true[ts_mask])
        
        loss_list.append(loss.detach())

    avg_loss = torch.mean(torch.stack(loss_list)).item()
    tr_pred_list = torch.cat(tr_pred_list, dim=-1)
    tr_true_list = torch.cat(tr_true_list, dim=-1)
    va_pred_list = torch.cat(va_pred_list, dim=-1)
    va_true_list = torch.cat(va_true_list, dim=-1)
    ts_pred_list = torch.cat(ts_pred_list, dim=-1)
    ts_true_list = torch.cat(ts_true_list, dim=-1)

    tr_acc = accuracy(tr_pred_list, tr_true_list).item()
    s_val_acc = accuracy(va_pred_list, va_true_list).item()
    s_test_acc = accuracy(ts_pred_list, ts_true_list).item()
    current_lr = lr_scheduler.get_last_lr()[0]
    
    return avg_loss, tr_acc, s_val_acc, s_test_acc, current_lr



@torch.no_grad()
def test(loader):
    model.eval()
    pred_list = []
    true_list = []
    for batch in tqdm(loader, dynamic_ncols=True):
        batch = batch.to(device, 'edge_index')
        batch_size = batch[targe_node_type].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)[:batch_size]
        pred = out.argmax(dim=-1)
        
        pred_list.append(pred)
        true_list.append(batch[targe_node_type][label_name][:batch_size])
    pred_list = torch.cat(pred_list, dim=-1)
    true_list = torch.cat(true_list, dim=-1)
    acc = accuracy(pred_list, true_list).item()
    return acc, f1_score(pred_list, true_list, average='macro', num_classes=num_classes)


@torch.no_grad()
def init_params():
    # Initialize lazy parameters via forwarding a single batch to the model:
    batch = next(iter(train_loader))
    batch = batch.to(device, 'edge_index')
    model(batch.x_dict, batch.edge_index_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train SeqHGNN')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=int, default=0, help='gpu device')
    parser.add_argument('--dataset', choices=['ogbn-mag', 'ogbn-mag-own', 'ogbn-mag-multi', 'ogbn-mag-complex', 'ogbn-mag-complex2', 'oag-field', 'oag-venue', 'oag-venue-multi', 'recruit', 'dblp', 'acm', 'imdb'], default='ogbn-mag', help='dataset name')
    parser.add_argument('--num-hidden', type=int, default=256, help='number of hidden units')
    parser.add_argument('--num-heads', type=int, default=8, help='num of attention heads')
    parser.add_argument('--num-layers', type=int, default=3, help='number of hidden layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--epochs', type=int, default=100, help='traninig epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='batch size')
    parser.add_argument('--n-batch', type=int, help='train batch num')
    parser.add_argument('--num-samples', type=int, help='number of neighbor samples')
    parser.add_argument("--amp", action='store_true', default=False, help="whether to amp to accelerate training with float16(half) calculation")
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--logsubfix', type=str, default='', help='log file subfix')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient Norm Clipping')
    parser.add_argument('--workers', type=int, default=0, help='number of data loader workers') 
    parser.add_argument('--save-model', default=False, action='store_true', help='if save model')
    parser.add_argument('--save-path', help='model save path', default='model/mpt_')
    parser.add_argument('--off-wandb', default=False, action='store_true', help='if use wandb')
    args = parser.parse_args()
    tags = [t for t in args.logsubfix.split(',')]
    wandb.init(project='SEQ_HGNN', entity='nobrowning', tags=tags, mode="disabled" if args.off_wandb else None)
    wandb.config.update(args)
    config = wandb.config
    seed_everything(args.seed)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler("seq_hgnn_{}_lr{}_e{}_drop{}.log".format(
                                args.logsubfix, args.lr, args.epochs, args.dropout)),
                            logging.StreamHandler()
                        ],
                        level=logging.INFO)
    logging.info(args)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    data, targe_node_type, label_name, num_classes, threshold = \
        load_data(dataset_name=args.dataset)
    data[targe_node_type].update({'ids': torch.arange(data[targe_node_type].x.shape[0])})
    data = data.to(device, 'x', label_name, 'ids')
    train_input_nodes = (targe_node_type, data[targe_node_type].train_mask)
    val_input_nodes = (targe_node_type, data[targe_node_type].val_mask)
    test_input_nodes = (targe_node_type, data[targe_node_type].test_mask)


    dl_kwargs = {'batch_size': args.batch_size, 'num_workers': args.workers,
               'num_samples': [args.num_samples] * (args.num_layers),
              'persistent_workers': True if args.workers > 0 else False, 'pin_memory': True}
    if args.num_samples is None:
        args.num_samples = 6*args.batch_size
    num_samples = []
    train_loader = HGTLoader(data, shuffle=True, input_nodes=train_input_nodes,
                             **dl_kwargs)
    val_loader = HGTLoader(data, input_nodes=val_input_nodes, **dl_kwargs)
    test_loader = HGTLoader(data, input_nodes=test_input_nodes, **dl_kwargs)
    model = SeqHGNN(graph_meta=data.metadata(), targe_node_type=targe_node_type, hidden_channels=args.num_hidden, out_channels=num_classes, num_heads=args.num_heads, num_layers=args.num_layers, dropout=args.dropout).to(device)
    
    init_params()  # Initialize parameters.
    wandb.watch(model)
    
    if args.amp:
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None
    
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
        if (train_acc >= threshold and s_val_acc > best_s_val_acc) or epoch == args.epochs - 1 or epoch == 0: # epoch % args.eval_every == 0
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

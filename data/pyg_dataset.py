from torch_geometric.data import HeteroData
from dgl.data.utils import load_graphs
import torch
import os
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import OGB_MAG, IMDB
import torch_geometric.transforms as T
import pickle


def load_data(dataset_name='ogbn-mag-complex', target_node='paper', cls_level=3, preprocess=None):
    transform = T.ToUndirected(merge=True)
    if dataset_name == 'ogbn-mag':
        dataset = OGB_MAG('D:\\Data\\graph\\MAG', preprocess='metapath2vec', transform=transform)
        targe_node_type = 'paper'
        num_classes = dataset.num_classes
        threshold = 0.5
        label_name = 'y'
    elif dataset_name == 'ogbn-mag-complex':
        dataset = MAGComplexDataset('dataset/OGB_MAG_COMPLEX/')
        targe_node_type = 'paper'
        num_classes = dataset.num_classes
        threshold = 0.5
        label_name = 'y'
    elif dataset_name == 'ogbn-mag-own':
        dataset = MAGDataset('dataset/OGB_MAG/')
        targe_node_type = 'paper'
        num_classes = dataset.num_classes
        threshold = 0.5
        label_name = 'y'
    elif dataset_name == 'ogbn-mag-multi':
        dataset = MAGDatasetMulti('dataset/OGB_MAG/')
        targe_node_type = 'paper'
        num_classes = dataset.num_classes
        threshold = 0.5
        label_name = 'y'
    elif dataset_name == 'oag-venue':
        dataset = OAGDataset('dataset/OAG_VENUE/')
        targe_node_type = 'paper'
        num_classes = dataset.num_classes
        threshold = 0.9
        label_name = 'y'
    elif dataset_name == 'oag-venue-multi':
        dataset = OAGDatasetMulti('dataset/OAG_VENUE/')
        targe_node_type = 'paper'
        num_classes = dataset.num_classes
        threshold = 0.9
        label_name = 'y'
    elif dataset_name == 'imdb':
        dataset = IMDB('/home/xxx/pyg_data/IMDB', transform=transform)
        targe_node_type = 'movie'
        num_classes = 3
        label_name = 'y'
    elif dataset_name == 'hgb-acm':
        targe_node_type = 'P'
        dataset = HGBDataset('dataset/HGB_ACM/', preprocess=preprocess,
                             target_ntype=targe_node_type)
        # dataset = HgbACMDataset('dataset/HGB_ACM/')
        num_classes = dataset.num_classes
        threshold = 0.9
        label_name = 'y'
    elif dataset_name == 'hgb-dblp':
        targe_node_type = 'A'
        dataset = HGBDataset('dataset/HGB_DBLP/', preprocess=preprocess,
                             target_ntype=targe_node_type)
        num_classes = dataset.num_classes
        threshold = 0.9
        label_name = 'y'
    elif dataset_name == 'hgb-imdb':
        targe_node_type = 'M'
        dataset = HGBDataset('dataset/HGB_IMDB/', preprocess=preprocess,
                             target_ntype=targe_node_type)
        num_classes = dataset.num_classes
        threshold = 0.55
        label_name = 'y'
    data = dataset[0]
    return data, targe_node_type, label_name, num_classes, threshold


class MAGDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.join(self.root, 'OGB_MAG.pkl'), 
                os.path.join(self.root, 'OGB_MAG_split_idx.pkl')]
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'own_mag', 'processed')
    
    @property
    def num_classes(self) -> int:
        return int(self.data['paper'].y.max()) + 1
    
    @property
    def processed_file_names(self):
        return 'mag.pt'


    def process(self):
        glist, label_dict = load_graphs("dataset/OGB_MAG/OGB_MAG.pkl")
        split = torch.load('dataset/OGB_MAG/OGB_MAG_split_idx.pkl')
        dgl_g = glist[0]
        data = HeteroData()
        
        for ntype in dgl_g.ntypes:
            data[ntype].x = dgl_g.nodes[ntype].data['feat']
        data['paper'].year = dgl_g.nodes['paper'].data['year'].reshape(-1)
        data['paper'].y = label_dict['paper'].reshape(-1)
        
        train_mask = torch.zeros(data['paper'].num_nodes).bool()
        val_mask = torch.zeros(data['paper'].num_nodes).bool()
        test_mask = torch.zeros(data['paper'].num_nodes).bool()

        train_mask[split['train']['paper']] = True
        val_mask[split['valid']['paper']] = True
        test_mask[split['test']['paper']] = True
        
        data['paper'].train_mask = train_mask
        data['paper'].val_mask = val_mask
        data['paper'].test_mask = test_mask

        for etype in dgl_g.canonical_etypes:
            u, v = dgl_g[etype].edges()
            edge_idx = torch.vstack((u, v))
            data[etype].edge_index = edge_idx
        torch.save(self.collate([data]), self.processed_paths[0])

class MAGComplexDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.join(self.root, 'OGB_MAG.pkl'), 
                os.path.join(self.root, 'OGB_MAG_split_idx.pkl')]
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'mag_complex', 'processed')
    
    @property
    def num_classes(self) -> int:
        return int(self.data['paper'].y.max()) + 1
    
    @property
    def processed_file_names(self):
        return 'mag.pt'


    def process(self):
        glist, label_dict = load_graphs(self.raw_file_names[0])
        split = torch.load(self.raw_file_names[1])
        dgl_g = glist[0]
        data = HeteroData()
        
        for ntype in dgl_g.ntypes:
            data[ntype].x = dgl_g.nodes[ntype].data['feat']
        data['paper'].year = dgl_g.nodes['paper'].data['year'].reshape(-1)
        data['paper'].y = label_dict['paper'].reshape(-1)
        
        train_mask = torch.zeros(data['paper'].num_nodes).bool()
        val_mask = torch.zeros(data['paper'].num_nodes).bool()
        test_mask = torch.zeros(data['paper'].num_nodes).bool()

        train_mask[split['train']['paper']] = True
        val_mask[split['valid']['paper']] = True
        test_mask[split['test']['paper']] = True
        
        data['paper'].train_mask = train_mask
        data['paper'].val_mask = val_mask
        data['paper'].test_mask = test_mask

        for etype in dgl_g.canonical_etypes:
            u, v = dgl_g[etype].edges()
            edge_idx = torch.vstack((u, v))
            data[etype].edge_index = edge_idx
        transform = T.ToUndirected(merge=True)
        data = transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])


class MAGDatasetMulti(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.join(self.root, 'OGB_MAG.pkl'), 
                os.path.join(self.root, 'OGB_MAG_split_idx.pkl')]
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'own_mag_multi', 'processed')
    
    @property
    def num_classes(self) -> int:
        return int(self.data['paper'].y.max()) + 1
    
    @property
    def processed_file_names(self):
        return 'mag.pt'


    def process(self):
        glist, label_dict = load_graphs("dataset/OGB_MAG/OGB_MAG.pkl")
        split = torch.load('dataset/OGB_MAG/OGB_MAG_split_idx.pkl')
        dgl_g = glist[0]
        data = HeteroData()
        
        for ntype in dgl_g.ntypes:
            data[ntype].x = dgl_g.nodes[ntype].data['feat'].reshape(dgl_g.num_nodes(ntype), -1, 128)
        data['paper'].year = dgl_g.nodes['paper'].data['year'].reshape(-1)
        data['paper'].y = label_dict['paper'].reshape(-1)
        
        train_mask = torch.zeros(data['paper'].num_nodes).bool()
        val_mask = torch.zeros(data['paper'].num_nodes).bool()
        test_mask = torch.zeros(data['paper'].num_nodes).bool()

        train_mask[split['train']['paper']] = True
        val_mask[split['valid']['paper']] = True
        test_mask[split['test']['paper']] = True
        
        data['paper'].train_mask = train_mask
        data['paper'].val_mask = val_mask
        data['paper'].test_mask = test_mask

        for etype in dgl_g.canonical_etypes:
            u, v = dgl_g[etype].edges()
            edge_idx = torch.vstack((u, v))
            data[etype].edge_index = edge_idx
        torch.save(self.collate([data]), self.processed_paths[0])


class OAGDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.join(self.root, 'graph.pkl')]
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'oag_venue', 'processed')
    
    @property
    def num_classes(self) -> int:
        return int(self.data['paper'].y.max()) + 1
    
    @property
    def processed_file_names(self):
        return 'oag_venue.pt'


    def process(self):
        glist, _ = load_graphs(self.raw_file_names[0])
        dgl_g = glist[0]
        data = HeteroData()

        for ntype in dgl_g.ntypes:
            data[ntype].x = dgl_g.nodes[ntype].data['feat']
        data['paper'].year = dgl_g.nodes['paper'].data['year'].reshape(-1)
        data['paper'].y = dgl_g.nodes['paper'].data['label'].reshape(-1)


        data['paper'].train_mask = dgl_g.nodes['paper'].data['train_mask'].bool()
        data['paper'].val_mask = dgl_g.nodes['paper'].data['val_mask'].bool()
        data['paper'].test_mask = dgl_g.nodes['paper'].data['test_mask'].bool()

        for etype in dgl_g.canonical_etypes:
            u, v = dgl_g[etype].edges()
            edge_idx = torch.vstack((u, v))
            data[etype].edge_index = edge_idx
        torch.save(self.collate([data]), self.processed_paths[0])


class OAGDatasetMulti(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.join(self.root, 'graph.pkl')]
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'oag_venue_multi', 'processed')
    
    @property
    def num_classes(self) -> int:
        return int(self.data['paper'].y.max()) + 1
    
    @property
    def processed_file_names(self):
        return 'oag_venue.pt'


    def process(self):
        glist, _ = load_graphs(self.raw_file_names[0])
        dgl_g = glist[0]
        data = HeteroData()

        for ntype in dgl_g.ntypes:
            data[ntype].x = dgl_g.nodes[ntype].data['feat'].reshape(dgl_g.num_nodes(ntype), -1, 128)
        data['paper'].year = dgl_g.nodes['paper'].data['year'].reshape(-1)
        data['paper'].y = dgl_g.nodes['paper'].data['label'].reshape(-1)


        data['paper'].train_mask = dgl_g.nodes['paper'].data['train_mask'].bool()
        data['paper'].val_mask = dgl_g.nodes['paper'].data['val_mask'].bool()
        data['paper'].test_mask = dgl_g.nodes['paper'].data['test_mask'].bool()

        for etype in dgl_g.canonical_etypes:
            u, v = dgl_g[etype].edges()
            edge_idx = torch.vstack((u, v))
            data[etype].edge_index = edge_idx
        torch.save(self.collate([data]), self.processed_paths[0])


class RecruitDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None,
                cls_level=2, target_node='jd'):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.cls_level = cls_level
        self.target_node = target_node

    @property
    def raw_file_names(self):
        return [os.path.join(self.root, 'hetero_graph2.bin')]
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'recruit_pyg', 'processed')
    
    @property
    def num_classes(self) -> int:
        return int(self.data[self.target_node]['y{}'.format(self.cls_level)].max()) + 1
    
    @property
    def neg_resume_dict(self) -> dict:
        path = os.path.join(self.root, 'neg_resume_dict.pkl')
        with open(path, 'rb') as f:
            neg_resume_dict = pickle.load(f)
        return neg_resume_dict
    
    @property
    def processed_file_names(self):
        return 'recruit.pt'


    def process(self):
        glist, _ = load_graphs(self.raw_file_names[0])
        dgl_g = glist[0]
        data = HeteroData()

        for ntype in dgl_g.ntypes:
            for feat_name in dgl_g.nodes[ntype].data.keys():
                if feat_name == 'feat':
                    pyg_feat_name = 'x'
                else:
                    pyg_feat_name = feat_name
                feat = dgl_g.nodes[ntype].data[feat_name]
                if 'mask' in feat_name:
                    feat = feat.bool()
                data[ntype][pyg_feat_name] = feat

        for etype in dgl_g.canonical_etypes:
            u, v = dgl_g[etype].edges()
            edge_idx = torch.vstack((u, v))
            data[etype].edge_index = edge_idx

            for edata_key in dgl_g[etype].edata.keys():
                feat = dgl_g[etype].edata[edata_key]
                if 'mask' in edata_key:
                    feat = feat.bool()
                data[etype][edata_key] = feat
        torch.save(self.collate([data]), self.processed_paths[0])


class HGBDataset(InMemoryDataset):
    def __init__(self, root, preprocess=None,
                 transform=None, pre_transform=None, 
                 pre_filter=None, target_ntype='A'):
        self.target_ntype = target_ntype
        self.preprocess = preprocess
        super().__init__(root, transform, pre_transform, pre_filter)
        assert self.preprocess in [None, 'metapath2vec', 'transe', 'complex']
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        if self.preprocess is not None:
            embs = torch.load(self.raw_file_names[1])
            for ntype in self.data.node_types:
                self.data[ntype].emb = embs[ntype].float()

    @property
    def raw_file_names(self):
        file_names = []
        raw_data_file = [n for n in os.listdir(self.root) if '.pkl' in n][0]
        file_names.append(raw_data_file)
        if self.preprocess is not None:
            file_names.append(f'{self.preprocess}_emb.pt')
        return [os.path.join(self.root, f) for f in file_names]
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')
    
    @property
    def num_classes(self) -> int:
        if len(self.data[self.target_ntype].y.shape) == 1:
            return int(self.data[self.target_ntype].y.max()) + 1
        return self.data[self.target_ntype].y.shape[-1]
    
    @property
    def processed_file_names(self):
        if self.preprocess is None:
            return 'processed_graph.pt'
        return f'processed_graph_{self.preprocess}.pt'


    def process(self):
        glist, label_dict = load_graphs(self.raw_file_names[0])
        target_ntype = [key for key in label_dict.keys() if '_' not in key][0]
        dgl_g = glist[0]
        data = HeteroData()
        
        # if self.preprocess is None:
        #     for ntype in dgl_g.ntypes:
        #         data[ntype].x = dgl_g.nodes[ntype].data[ntype]
        # else:
        #     embs = torch.load(self.raw_file_names[1])
        #     for ntype in dgl_g.ntypes:
        #         feats = torch.cat([embs[ntype], dgl_g.nodes[ntype].data[ntype]], dim=1)
        #         data[ntype].x = feats
        
        for ntype in dgl_g.ntypes:
            data[ntype].x = dgl_g.nodes[ntype].data[ntype]
        
        # if self.preprocess is not None:
        #     embs = torch.load(self.raw_file_names[1])
        #     for ntype in dgl_g.ntypes:
        #         data[ntype].emb = embs[ntype]
        
        labels = label_dict[target_ntype]
        if len(labels.shape) == 2:
            labels = labels.float()
        data[target_ntype].y = labels
        train_mask = torch.zeros(data[target_ntype].num_nodes).bool()
        val_mask = torch.clone(train_mask)
        test_mask = torch.clone(train_mask)
        test_mask_full = torch.clone(train_mask)

        train_mask[label_dict[f'{target_ntype}_train']] = True
        val_mask[label_dict[f'{target_ntype}_val']] = True
        test_mask[label_dict[f'{target_ntype}_test']] = True
        test_mask_full[label_dict[f'{target_ntype}_test_full']] = True

        data[target_ntype].train_mask = train_mask
        data[target_ntype].val_mask = val_mask
        data[target_ntype].test_mask = test_mask
        data[target_ntype].test_mask_full = test_mask_full
        
        for etype in dgl_g.canonical_etypes:
            u, v = dgl_g[etype].edges()
            edge_idx = torch.vstack((u, v))
            data[etype].edge_index = edge_idx
        # transform = T.ToUndirected(merge=False)
        # data = transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])
import os
import torch
import numpy
import random
import itertools
from torch import Tensor, LongTensor
from tqdm import tqdm
import datetime
from ogb.linkproppred import PygLinkPropPredDataset
from typing import Literal, NamedTuple, List, Optional
from tap import Tap

from torch_geometric.data import Data
from torch_geometric.datasets import SNAPDataset, FacebookPagePage, WikipediaNetwork
from torch_geometric.utils import negative_sampling, to_undirected

import tools

import matplotlib.pyplot as plt



class Arguments(Tap):
    dataset: List[
        Literal[
            "ogbl-ddi",
            "ogbl-ppa",
            "ogbl-collab",
            "soc-epinions1",
            "soc-livejournal1",
            "soc-pokec",
            "soc-slashdot0811",
            "soc-slashdot0922",
            "facebook",
            "wikipedia",
        ]
    ]
    # the dataset to run the experiment on
    dataset_dir: str = "data"  # directory containing the dataset files
    method: List[
        Literal[
            "hyperlearn",
        ]
    ]
    # method to run the experiment with
    dimensions: List[int]
    # number of dimensions to use (does not affect the exact method)
    batch_size: int = 16384  # number of nodes to evaluate at once
    result_dir: str = "results"  # directory to write the results to
    device: List[str] = ["cpu"]  # which device to run the experiment on
    seed: List[int] = [1]  # random number generator seed
    lr: float = 0.1
    nitr: int = 0
    binarize: bool = False
    use_node_features: List[int] = [0]


class Config(NamedTuple):
    dataset: str  # the dataset to run the experiment on
    method: str  # method to run the experiment with
    dimensions: int  # number of dimensions to use (does not affect the exact method)
    device: torch.device  # which device to run the experiment on
    seed: int  # random number generator seed
    use_node_features: int


class Result(NamedTuple):
    output_pos: torch.Tensor
    output_neg: torch.Tensor
    init_time: float
    calc_time: float
    dimensions: int


METRICS = [
    "method",
    "dataset",
    "dimensions",
    "hits@20",
    "hits@50",
    "hits@100",
    "init_time",
    "calc_time",
    "num_node_pairs",
    "device",
    "nitr",
    "use_node_features",
]


class Method:
    signatures: Optional[Tensor]
    num_neighbors: Optional[Tensor]

    def __init__(self, dimensions, batch_size, device):
        self.dimensions = dimensions
        self.device = device
        self.batch_size = batch_size

    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        NotImplemented

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        NotImplemented


def find_thresh(pos_sim:Tensor,neg_sim:Tensor):
    pos_sim=sorted(pos_sim)
    neg_sim=sorted(neg_sim)

    total_neg=len(neg_sim)
    total_pos=len(pos_sim)


    current_pos_indx=0

    for n in range(total_neg):
        t=neg_sim[n]
        while pos_sim[current_pos_indx]<t and current_pos_indx<total_pos-1:
            current_pos_indx+=1

        if (total_neg-n)/total_neg<current_pos_indx/total_pos:
            print("Threshold:",t)
            print("False Negative Rate:",(total_neg-n)/total_neg)
            print("False Positive Rate:",current_pos_indx/total_pos)
            print(total_neg,total_pos,n,current_pos_indx)
            return t
        



class HyperLearnMethod(Method):
    def init_signatures(self, edge_index: LongTensor, edge_neg_index: LongTensor,num_nodes: int, nodeFeatures = None):

        if nodeFeatures is not None:

            print("Node Features is not None")
            randon_matrix=torch.randn(nodeFeatures.shape[1],self.dimensions,device=self.device)
            
            M=nodeFeatures@randon_matrix

            node_vectors = torch.complex(torch.cos(M),torch.sin(M))

            scale =(1 / self.dimensions)**0.5
            node_vectors.mul_(scale)  # make them unit vectors
            print(nodeFeatures)
            print("Node Features Encoded")
        else:
            node_vectors = tools.get_random_node_vectors(
                num_nodes, self.dimensions, device=self.device
            )
      

        self.node_vectors = node_vectors
        self.memory = tools.get_node_signatures(
            edge_index, node_vectors, self.batch_size
        ) -tools.get_node_signatures(
            edge_neg_index, node_vectors, self.batch_size
           )
       

        self.edge_index = edge_index
        self.edge_neg_index = edge_neg_index

        self.retrain_signatures(1,0.1)

        

        

    def scaling(self,edge_index: LongTensor, num_nodes: int):
        NotImplemented


    def retrain_signatures(self,  nitr: int=1,lr:float=0.1):

        
        node_vectors=self.node_vectors
        memory=self.memory
        edge_index=self.edge_index
        edge_neg_index=self.edge_neg_index


        #signatures=self.signatures

        for _ in range(nitr):
            self.retrain_signatures_itr(edge_index, edge_neg_index, node_vectors,memory,self.batch_size, lr)


    def retrain_signatures_itr(self,
        edge_index: LongTensor, edge_neg_index: LongTensor, node_vectors: Tensor, memory: Tensor,batch_size: int,
        lr:float=0.1, t:float=0.00 ):

        to_nodes, from_nodes = edge_index
        to_neg_nodes, from_neg_nodes=edge_neg_index

        pos_similarity=[]
        neg_similarity=[]


        sub_sample_size=min(10000,len(to_nodes),len(to_neg_nodes))

        sample=torch.randperm(len(to_nodes))[:sub_sample_size]
        to_batches = torch.split(to_nodes[sample], batch_size)  
        from_batches = torch.split(from_nodes[sample], batch_size)

        for to_batch, from_batch in zip(to_batches, from_batches):
            from_vec = torch.index_select(node_vectors, 0, from_batch)
            to_memory = torch.index_select(memory, 0, to_batch)

            pos_similarity.append(tools.cdot(to_memory,from_vec))

        sample=torch.randperm(len(to_neg_nodes))[:sub_sample_size]
        to_neg_batches = torch.split(to_neg_nodes[sample], batch_size)
        from_neg_batches = torch.split(from_neg_nodes[sample], batch_size)

        for to_neg_batch, from_neg_batch in zip(to_neg_batches, from_neg_batches):
            from_neg_vec = torch.index_select(node_vectors, 0, from_neg_batch)
            to_neg_memory = torch.index_select(memory, 0, to_neg_batch)

            neg_similarity.append(tools.cdot(to_neg_memory,from_neg_vec))

        pos_similarity = torch.cat(pos_similarity)
        neg_similarity = torch.cat(neg_similarity)

        
        #sub_sample_pos=pos_similarity[]
        #sub_sample_neg=neg_similarity[torch.randperm(len(neg_similarity))[:sub_sample_size]]


        T=find_thresh(pos_similarity,neg_similarity)
        print(T)




    

        #plt.hist(pos_similarity.cpu().detach().numpy(), bins=1000, alpha=0.5, label='pos')
        #plt.hist(neg_similarity.cpu().detach().numpy(), bins=1000, alpha=0.5, label='neg')
        #plt.legend(loc='upper right')
        #plt.show()

        to_batches = torch.split(to_nodes, batch_size)  
        from_batches = torch.split(from_nodes, batch_size)
        

        for to_batch, from_batch in zip(to_batches, from_batches):
            from_vec = torch.index_select(node_vectors, 0, from_batch)
            to_memory = torch.index_select(memory, 0, to_batch)
            sim_val= tools.cdot(to_memory,from_vec)
            mask=(sim_val<T).float()
            sign=-mask
            #sign=(sim_val-T)*mask #Remember square root is attached

            memory.index_add_(0, to_batch,-sign.unsqueeze(1)*lr*from_vec)


        to_neg_batches = torch.split(to_neg_nodes, batch_size)
        from_neg_batches = torch.split(from_neg_nodes, batch_size)

        for to_neg_batch, from_neg_batch in zip(to_neg_batches, from_neg_batches):
            
            from_neg_vec = torch.index_select(node_vectors, 0, from_neg_batch)
            to_neg_memory = torch.index_select(memory, 0, to_neg_batch)
            sim_val= tools.cdot(to_neg_memory,from_neg_vec)
            mask=(sim_val>T).float()

            sign=(sim_val-T)*mask

            memory.index_add_(0, to_neg_batch,-sign.unsqueeze(1)*lr*from_neg_vec)



    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        
        return tools.cdot(
            self.memory[node_ids],
            self.memory[other_ids],
        )







class LinkPredDataset:
    def __init__(self, graph: Data):
        self.graph = graph

        edge_index = graph.edge_index
        num_nodes = graph.num_nodes
        num_edges = graph.num_edges
        num_test_edges = num_edges // 20

        edge_neg = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            method="sparse",
            force_undirected=True,
            num_neg_samples=num_edges,
        )

        permuted_indices = torch.randperm(num_edges)
        self.edge = edge_index[:, permuted_indices[:num_test_edges]]
        self.data = Data(
            edge_index=edge_index[:, permuted_indices[num_test_edges:]],
            num_nodes=num_nodes,
            edge_neg_index=edge_neg[:,num_test_edges:],
        )

        self.edge_neg=edge_neg[:, :num_test_edges]

        try:
            self.x=graph.x
        except:
            self.x=None

    def get_edge_split(self):
        train = None
        valid = None

        test = {
            "edge": self.edge.T,
            "edge_neg": self.edge_neg.T,
        }

        return {"train": train, "valid": valid, "test": test}


def evaluate_hits_at(pred_pos: Tensor, pred_neg: Tensor, K: int) -> float:
    if len(pred_neg) < K:
        return 1.0

    kth_score_in_negative_edges = torch.topk(pred_neg, K)[0][-1]
    print(kth_score_in_negative_edges)
    num_hits = torch.sum(pred_pos > kth_score_in_negative_edges).cpu()
    hitsK = float(num_hits) / len(pred_pos)
    return hitsK


def executor(args: Arguments, method: Method, dataset, conf, retrain=False,device=None,write=None):
    graph = dataset.data.to(device)
    split_edge = dataset.get_edge_split()
    pos_test_edge = split_edge["test"]["edge"].to(device)
    neg_test_edge = split_edge["test"]["edge_neg"].to(device)

    get_duration = tools.stopwatch()
    method.init_signatures(to_undirected(graph.edge_index),to_undirected(graph.edge_neg_index), graph.num_nodes, nodeFeatures=dataset.x)
    init_time = get_duration()

    get_duration = tools.stopwatch()
    if retrain:

        
        pos_test_edge_loader = pos_test_edge.split(args.batch_size)
        neg_test_edge_loader = neg_test_edge.split(args.batch_size)
        for n in range(args.nitr):

            if write is not None:

                pos_scores = []
                neg_scores = []

                for edge_batch in tqdm(pos_test_edge_loader, leave=False):
                    node_ids, other_ids = edge_batch[:, 0], edge_batch[:, 1]
                    scores = method.calc_scores(node_ids, other_ids)
                    pos_scores.append(scores.cpu())

                for edge_batch in tqdm(neg_test_edge_loader, leave=False):
                    node_ids, other_ids = edge_batch[:, 0], edge_batch[:, 1]
                    scores = method.calc_scores(node_ids, other_ids)
                    neg_scores.append(scores.cpu())

                pos_scores = torch.cat(pos_scores)
                neg_scores = torch.cat(neg_scores)

                dimensions = method.node_vectors.shape[1]

                res= Result(pos_scores, neg_scores, -1, -1, dimensions)
                hits=get_hits(res)

                out={
                    "dimensions": res.dimensions,
                    "hits@20": hits[0],
                    "hits@50": hits[1],
                    "hits@100": hits[2],
                    "init_time": res.init_time,
                    "calc_time": res.calc_time,
                    "num_node_pairs": res.output_pos.size(0) + res.output_neg.size(0),
                }

                out["method"] = conf.method
                out["dataset"] = conf.dataset
                out["device"] = conf.device.type
                out["nitr"]=n
                out["use_node_features"] = conf.use_node_features
                write(out)



            method.retrain_signatures(nitr=1,lr=args.lr)

            
    retrain_time = get_duration()
    print("retrain time:",retrain_time)
    

    pos_scores = []
    neg_scores = []
    pos_test_edge_loader = pos_test_edge.split(args.batch_size)
    neg_test_edge_loader = neg_test_edge.split(args.batch_size)

    get_duration = tools.stopwatch()
    for edge_batch in tqdm(pos_test_edge_loader, leave=False):
        node_ids, other_ids = edge_batch[:, 0], edge_batch[:, 1]
        scores = method.calc_scores(node_ids, other_ids)
        pos_scores.append(scores.cpu())

    for edge_batch in tqdm(neg_test_edge_loader, leave=False):
        node_ids, other_ids = edge_batch[:, 0], edge_batch[:, 1]
        scores = method.calc_scores(node_ids, other_ids)
        neg_scores.append(scores.cpu())

    calc_time = get_duration()
    pos_scores = torch.cat(pos_scores)
    neg_scores = torch.cat(neg_scores)

    dimensions = method.node_vectors.shape[1]

    return Result(pos_scores, neg_scores, init_time, calc_time, dimensions)


def get_dataset(name: str, root: str):
    if name.startswith("ogbl-"):
        dataset= PygLinkPropPredDataset(name, root)
        return LinkPredDataset(dataset.data)
    elif name.startswith("soc-"):
        dataset = SNAPDataset(root, name)
        return LinkPredDataset(dataset.data)
    elif name == "facebook":
        dataset = FacebookPagePage(os.path.join(root, "facebookpagepage"))
        return LinkPredDataset(dataset.data)
    elif name == "wikipedia":
        dataset = WikipediaNetwork("data", "crocodile", geom_gcn_preprocess=False)
        return LinkPredDataset(dataset.data)
    else:
        raise NotImplementedError()


def get_hits(result: Result):
    output = []

    for K in [20, 50, 100]:
        test_hits = evaluate_hits_at(result.output_pos, result.output_neg, K)
        output.append(test_hits)

    return tuple(output)


def get_metrics(conf: Config, args: Arguments, dataset, device=None,write=None):

    retrain=False
   
    if conf.method == "hyperlearn":
        method_cls = HyperLearnMethod
        retrain=True
    else:
        raise NotImplementedError(f"requested method {conf.method} is not implemented")

    method = method_cls(conf.dimensions, args.batch_size, device)

    result = executor(args, method, dataset,conf,retrain=retrain, device=device,write=write)
    #print("Shape out: ", result.output_pos.shape, result.output_neg.shape)

    #plt.hist(result.output_pos.cpu().detach().numpy(), bins=1000, alpha=0.5, label='pos')
    #plt.hist(result.output_neg.cpu().detach().numpy(), bins=1000, alpha=0.5, label='neg')
    #plt.legend(loc='upper right')
    #plt.show()

    total_time = result.init_time + result.calc_time
    num_node_pairs = result.output_pos.size(0) + result.output_neg.size(0)
    hits = get_hits(result)
    print(
        f"{conf.method}: Hits: {hits[0]:.3g}@20, {hits[1]:.3g}@50, {hits[2]:.3g}@100; Time: {result.init_time:.3g}s + {result.calc_time:.3g}s = {total_time:.3g}s; Dims: {result.dimensions}"
    )

    return {
        "dimensions": result.dimensions,
        "hits@20": hits[0],
        "hits@50": hits[1],
        "hits@100": hits[2],
        "init_time": result.init_time,
        "calc_time": result.calc_time,
        "num_node_pairs": num_node_pairs,
    }


def main(conf: Config, args: Arguments, result_file: str):

    torch.manual_seed(conf.seed)
    numpy.random.seed(conf.seed)
    random.seed(conf.seed)

    print("Device:", conf.device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with tools.open_metric_writer(result_file, METRICS) as write:

        print("Dataset:", conf.dataset)
        dataset = get_dataset(conf.dataset, args.dataset_dir)

        if not conf.use_node_features:
            dataset.x = None
            print("Not using Node Features")
        else:
            try:
                a=dataset.x
                print("Node Features detected")
                print(a)
                print("Node Features in the dataset reader")
            except:
                print("No Node Features detected")
                return -1
            
            if dataset.x is None:
                print("No Node Features detected")
                return -1

        try:
            metrics = get_metrics(conf, args, dataset, device=conf.device,write=write)
            metrics["method"] = conf.method
            metrics["dataset"] = conf.dataset
            metrics["device"] = conf.device.type
            metrics["use_node_features"] = conf.use_node_features
            metrics["nitr"]=args.nitr
            write(metrics)
        except Exception as e:
            print(e)


def default_to_cpu(device: str) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(device)
    else:
        return torch.device("cpu")


if __name__ == "__main__":



    args = Arguments(underscores_to_dashes=True).parse_args()

    result_filename = (
        "link_prediction-learned-"
        + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        + ".csv"
    )

    result_file = os.path.join(args.result_dir, result_filename)
    os.makedirs(args.result_dir, exist_ok=True)

    devices = {default_to_cpu(d) for d in args.device}

    options = (args.seed, devices, args.dimensions, args.dataset, args.method, args.use_node_features)
    for seed, device, dimensions, dataset, method, use_node_features in itertools.product(*options):
        config = Config(dataset, method, dimensions, device, seed, use_node_features)
        main(config, args, result_file)

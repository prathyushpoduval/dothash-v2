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


class Arguments(Tap):
    dataset: List[
        Literal[
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
            "hyperhash-adamic-adar",
            "hyperhash-common-neighbors",
            "hyperhash-common-neighbors",
            "hyperhash-resource-allocation",
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


class Config(NamedTuple):
    dataset: str  # the dataset to run the experiment on
    method: str  # method to run the experiment with
    dimensions: int  # number of dimensions to use (does not affect the exact method)
    device: torch.device  # which device to run the experiment on
    seed: int  # random number generator seed


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



class HyperLearnMethod(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int, nodeFeatures = None):

        if nodeFeatures is not None:
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
      
        node_scaling = self.scaling(edge_index, num_nodes)**2
        node_vectors_scaled=node_scaling.unsqueeze(1)*node_vectors

        self.node_vectors = node_vectors
        self.node_vectors_scaled = node_vectors_scaled
        self.node_scaling = node_scaling
        self.edge_index = edge_index

        self.signatures_scaled = tools.get_node_signatures(
            edge_index, node_vectors_scaled, self.batch_size
        )*0
        print("signatures scaled")

        self.signatures = tools.get_node_signatures(edge_index, node_vectors, self.batch_size)
        print("signatures")
        print("retrained")
        

        

        

    def scaling(self,edge_index: LongTensor, num_nodes: int):
        NotImplemented


    def retrain_signatures(self,  nitr: int=1,lr:float=0.1):

        
        node_scaling=self.node_scaling
        node_vectors_scaled=self.node_vectors_scaled
        node_vectors=self.node_vectors
        edge_index=self.edge_index


        #signatures=self.signatures
        signatures_scaled=self.signatures_scaled

        for _ in range(nitr):
            self.retrain_signatures_itr(edge_index, node_vectors,node_vectors_scaled,self.batch_size, node_scaling, signatures_scaled, lr)


    def retrain_signatures_itr(self,
        edge_index: LongTensor, node_vectors: Tensor, node_vectors_scaled: Tensor,batch_size: int,
        from_scaling_list: Tensor, signatures_scaled:Tensor, lr:float=0.1, t:float=0.00 ):

        to_nodes, from_nodes = edge_index
        
        to_batches = torch.split(to_nodes, batch_size)  
        from_batches = torch.split(from_nodes, batch_size)

        
        dimensions=node_vectors.shape[1]

        for to_batch, from_batch in zip(to_batches, from_batches):
            from_node_vectors = torch.index_select(node_vectors, 0, from_batch)
            from_node_vectors_scaled = torch.index_select(node_vectors_scaled, 0, from_batch)
            from_scaling=torch.index_select(from_scaling_list, 0, from_batch)
            to_signatures_scaled=signatures_scaled.index_select(0, to_batch)

            from_scaling_estimate=torch.real(tools.cdot( to_signatures_scaled,from_node_vectors))

            print(from_scaling_estimate)
            print(from_scaling)

            sign=(from_scaling_estimate-from_scaling) #Remember square root is attached

            mask=(torch.abs(sign)>t).float()
            sign=sign*mask

            signatures_scaled.index_add_(0, to_batch,-sign.unsqueeze(1)*lr*from_node_vectors_scaled)



    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        
        return tools.cdot(
            self.signatures_scaled[node_ids],
            self.signatures[other_ids],
        )

class HyperHashAdamicAdar(HyperLearnMethod):
    def scaling(self,edge_index: LongTensor, num_nodes: int):
        node_scaling = tools.get_adamic_adar_node_scaling(edge_index, num_nodes)
        return node_scaling


class HyperHashCommonNeighbours(HyperLearnMethod):
    def scaling(self,edge_index: LongTensor, num_nodes: int):
        return torch.zeros(num_nodes, device=self.device)+1

class HyperHashResourceAllocation(HyperLearnMethod):
   
    def scaling(self,edge_index: LongTensor, num_nodes: int):
        return tools.get_resource_allocation_node_scaling(edge_index, num_nodes)







class LinkPredDataset:
    def __init__(self, graph: Data):
        self.graph = graph

        edge_index = graph.edge_index
        num_nodes = graph.num_nodes
        num_edges = graph.num_edges
        num_test_edges = num_edges // 20

        self.edge_neg = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            method="sparse",
            force_undirected=True,
            num_neg_samples=num_test_edges,
        )

        permuted_indices = torch.randperm(num_edges)
        self.edge = edge_index[:, permuted_indices[:num_test_edges]]
        self.data = Data(
            edge_index=edge_index[:, permuted_indices[num_test_edges:]],
            num_nodes=num_nodes,
        )

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
    num_hits = torch.sum(pred_pos > kth_score_in_negative_edges).cpu()
    hitsK = float(num_hits) / len(pred_pos)
    return hitsK


def executor(args: Arguments, method: Method, dataset, retrain=False,device=None):
    graph = dataset.data.to(device)
    split_edge = dataset.get_edge_split()
    pos_test_edge = split_edge["test"]["edge"].to(device)
    neg_test_edge = split_edge["test"]["edge_neg"].to(device)

    get_duration = tools.stopwatch()
    method.init_signatures(to_undirected(graph.edge_index), graph.num_nodes, nodeFeatures=dataset.x)
    init_time = get_duration()

    get_duration = tools.stopwatch()
    if retrain:
        method.retrain_signatures(nitr=args.nitr,lr=args.lr)
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

    dimensions = method.signatures.shape[1]

    return Result(pos_scores, neg_scores, init_time, calc_time, dimensions)


def get_dataset(name: str, root: str):
    if name.startswith("ogbl-"):
        return PygLinkPropPredDataset(name, root)
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


def get_metrics(conf: Config, args: Arguments, dataset, device=None):

    retrain=False
   
    if conf.method == "hyperhash-common-neighbors":
        method_cls = HyperHashCommonNeighbours
        retrain=True
    elif conf.method == "hyperhash-adamic-adar":
        method_cls = HyperHashAdamicAdar
        retrain=True
    elif conf.method == "hyperhash-common-neighbors":
        method_cls = HyperHashCommonNeighbours
        retrain=True
    elif conf.method == "hyperhash-resource-allocation":
        method_cls = HyperHashResourceAllocation
        retrain=True
    else:
        raise NotImplementedError(f"requested method {conf.method} is not implemented")

    method = method_cls(conf.dimensions, args.batch_size, device)

    result = executor(args, method, dataset,retrain=retrain, device=device)
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
        try:
            a=dataset.x
            print("Node Features detected")
        except:
            print("No Node Features detected")
            return -1

        try:
            metrics = get_metrics(conf, args, dataset, device=conf.device)
            metrics["method"] = conf.method
            metrics["dataset"] = conf.dataset
            metrics["device"] = conf.device.type
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

    options = (args.seed, devices, args.dimensions, args.dataset, args.method)
    for seed, device, dimensions, dataset, method in itertools.product(*options):
        config = Config(dataset, method, dimensions, device, seed)
        main(config, args, result_file)

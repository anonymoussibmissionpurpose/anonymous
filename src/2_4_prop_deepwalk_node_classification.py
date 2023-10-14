import argparse

from tqdm import tqdm
from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes
import os
from torch_geometric.datasets import (
    Planetoid,
    Amazon,
    Flickr,
    CitationFull,
    CoraFull,
    Reddit,
    Coauthor,
)

from utils import (
    load_data,
    load_data_social,
    accuracy,
    mask_test_edges,
    preprocess_graph,
    loss_function_gcn,
    get_roc_score_GCN,
    torch_sparse_tensor_to_sparse_mx,
    do_transductive_edge_split,
)
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx
import torch.nn.functional as F

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from scipy.sparse import issparse
import argparse
import scipy.sparse as sp
import torch.nn as nn
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from os import path

from torch_sparse import SparseTensor
from typing import Union

from torch_geometric.utils import negative_sampling

from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder


from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


import warnings

warnings.filterwarnings("ignore")


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=bool, default=True, help="Enable CUDA training.")
parser.add_argument(
    "--dataset",
    type=str,
    default="Amazon-Computers",
    help="One dataset from \
Cora, CiteSeer, PubMed, CoraFull, Amazon-Computers, Amazon-Photo, \
CitationFull-DBLP, Flickr, Reddit, Coauthor-CS, Coauthor-Physics.",
)

parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--epochs", type=int, default=400, help="Number of epochs to train."  # 400
)
parser.add_argument("--p", type=float, default=1.0, help="Initial learning rate.")
parser.add_argument("--q", type=float, default=1.0, help="Initial learning rate.")

parser.add_argument(
    "--epoch_threshold", type=int, default=300, help="Dimension of embeddings."  # 300
)


parser.add_argument(
    "--num_of_nodes", type=int, default=-1, help="Write reuslts in files."
)

parser.add_argument(
    "--rank",
    type=int,
    default=32,
    help="The bound of the rank for the learnable embedding matrix.",
)

parser.add_argument(
    "--embedding_dim", type=int, default=32, help="Dimension of embeddings."
)

parser.add_argument(
    "--lr",
    type=float,
    default=0.01,
    help="The bound of the rank for the learnable embedding matrix.",
)


parser.add_argument("--write", type=int, default=1, help="Write reuslts in files.")

parser.add_argument("--cluster", type=int, default=0, help="Dimension of embeddings.")

parser.add_argument("--comp_idx", type=int, default=-1, help="Dimension of embeddings.")


parser.add_argument(
    "--feature_dim",
    type=float,
    default=0.0,
    help="The bound of the rank for the learnable embedding matrix.",
)


args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
dataset_name = args.dataset
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = "cpu"
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = "cuda"


def index2ptr(index: Tensor, size: int) -> Tensor:
    return torch._convert_indices_from_coo_to_csr(
        index, size, out_int32=index.dtype == torch.int32
    )


class MlpProdDecoder(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, class_num):
        super().__init__()

        self.embedding_size = embedding_size
        self.net = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, class_num),
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))


class SimpleMessagePassing(MessagePassing):
    def __init__(self):
        super(SimpleMessagePassing, self).__init__(aggr="mean")

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


class Node2Vec(torch.nn.Module):
    def __init__(
        self,
        edge_index: Tensor,
        embedding_dim: int,
        walk_length: int,
        context_size: int,
        rank_bound: int = 16,
        walks_per_node: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        num_nodes: Optional[int] = None,
        sparse: bool = False,
    ):
        super().__init__()

        if p == 1.0 and q == 1.0:
            self.random_walk_fn = torch.ops.pyg.random_walk
        else:
            self.random_walk_fn = torch.ops.torch_cluster.random_walk

        self.num_nodes = maybe_num_nodes(edge_index, num_nodes)

        row, col = sort_edge_index(edge_index, num_nodes=self.num_nodes).cpu()
        self.rowptr, self.col = index2ptr(row, self.num_nodes), col

        self.EPS = 1e-15
        assert walk_length >= context_size

        self.rank_bound = rank_bound

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples

        self.U = Embedding(self.num_nodes, self.rank_bound).cuda()
        self.V = Embedding(self.rank_bound, self.embedding_dim).cuda()

        self.model_message_passing = SimpleMessagePassing().cuda()
        self.edge_index = edge_index.cuda()

        self.reset_parameters()

    def reset_parameters(self):
        self.U.reset_parameters()
        transition_matrix_v = (torch.eye(args.rank, args.embedding_dim)).to(device)
        self.V.weight = torch.nn.Parameter(transition_matrix_v)
        self.V.weight.requires_grad = False

    def forward(self, batch: Optional[Tensor] = None, test=0) -> Tensor:
        embeddings = torch.matmul(self.U.weight, self.V.weight)

        embeddings = self.model_message_passing(embeddings, self.edge_index)

        return embeddings if batch is None else embeddings.index_select(0, batch)

    def loader(self, **kwargs) -> DataLoader:
        return DataLoader(range(self.num_nodes), collate_fn=self.sample, **kwargs)

    @torch.jit.export
    def pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)
        rw = self.random_walk_fn(
            self.rowptr, self.col, batch, self.walk_length, self.p, self.q
        )
        if not isinstance(rw, Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j : j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = torch.randint(
            self.num_nodes,
            (batch.size(0), self.walk_length),
            dtype=batch.dtype,
            device=batch.device,
        )
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j : j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def sample(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    @torch.jit.export
    def loss(self, pos_rw: Tensor, neg_rw: Tensor, epoch) -> Tensor:
        embeddings = torch.matmul(self.U.weight, self.V.weight)

        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = embeddings[start].view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = embeddings[rest.view(-1)].view(pos_rw.size(0), -1, self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = embeddings[start].view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = embeddings[rest.view(-1)].view(neg_rw.size(0), -1, self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss

    def test(
        self,
        train_z: Tensor,
        train_y: Tensor,
        test_z: Tensor,
        test_y: Tensor,
        solver: str = "lbfgs",  # lbfgs
        multi_class: str = "auto",
        *args,
        **kwargs,
    ) -> float:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            solver=solver, multi_class=multi_class, *args, **kwargs
        ).fit(train_z.detach().cpu().numpy(), train_y.detach().cpu().numpy())
        macro_f1 = f1_score(
            test_y.detach().cpu().numpy(),
            clf.predict(test_z.detach().cpu().numpy()),
            average="macro",
        )
        micro_f1 = f1_score(
            test_y.detach().cpu().numpy(),
            clf.predict(test_z.detach().cpu().numpy()),
            average="micro",
        )
        auc = roc_auc_score(
            test_y.detach().cpu().numpy(),
            clf.predict_proba(test_z.detach().cpu().numpy()),
            multi_class="ovr",
        )

        return (
            clf.score(test_z.detach().cpu().numpy(), test_y.detach().cpu().numpy()),
            macro_f1,
            micro_f1,
            auc,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.embedding.size(0)}, "
            f"{self.embedding.size(1)})"
        )


def get_roc_score(emb, edges_pos, edges_neg):
    def sigmoid(x):
        x = x.astype(np.float128)
        return 1 / (1 + np.exp(-x))

    preds = []

    for e in edges_pos:
        preds.append(sigmoid((emb[e[0], :] * emb[e[1], :]).sum()))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid((emb[e[0], :] * emb[e[1], :]).sum()))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score, preds_all


def scipy_sparse_to_torch_tensor(sparse_matrix):
    if not issparse(sparse_matrix):
        raise ValueError("Input matrix should be a SciPy sparse matrix.")

    coo_matrix = sparse_matrix.tocoo()
    indices = torch.tensor([coo_matrix.row, coo_matrix.col], dtype=torch.long)

    return indices


def build_graph_with_max_nodes(edges, max_nodes):
    edgelist = [tuple(edge) for edge in edges.cpu().numpy().T]

    G = nx.from_edgelist(edgelist)

    num_nodes = G.number_of_nodes()

    existing_nodes = set(G.nodes())
    missing_nodes = set(range(max_nodes)) - existing_nodes

    for node in missing_nodes:
        G.add_node(node)

    return G


def dataset_split(data):
    # get the nodes
    nodes = data.edge_index.t().numpy()
    nodes = np.unique(list(nodes[:, 0]) + list(nodes[:, 1]))

    total_num = nodes.max() + 1

    np.random.shuffle(nodes)  # shuffle node order

    # get train test and val sizes: (60% - 20% - 20%)
    train_size = int(total_num * 0.6)
    test_size = int(total_num * 0.8) - train_size
    val_size = total_num - train_size - test_size

    # get train test and validation set of nodes
    train_set = nodes[0:train_size]
    test_set = nodes[train_size : train_size + test_size]
    val_set = nodes[train_size + test_size :]

    train_mask = torch.zeros(total_num, dtype=torch.bool, device=device)
    for i in train_set:
        train_mask[i] = True

    test_mask = torch.zeros(total_num, dtype=torch.bool, device=device)
    for i in test_set:
        test_mask[i] = True

    val_mask = torch.zeros(total_num, dtype=torch.bool, device=device)
    for i in val_set:
        val_mask[i] = True

    # add masks
    data.train_mask = train_mask.cpu().numpy()
    data.test_mask = test_mask.cpu().numpy()
    data.val_mask = val_mask.cpu().numpy()

    return data


if __name__ == "__main__":
    # load data with args

    assert args.embedding_dim >= args.rank

    dataset = None
    data = None

    print("Loading dataset ... ")
    if args.dataset == "Cora":
        dataset = Planetoid(root="../data/Cora", name="Cora")
        data = dataset[0]
    elif args.dataset == "CiteSeer":
        dataset = Planetoid(root="../data/CiteSeer", name="CiteSeer")
        data = dataset[0]
    elif args.dataset == "PubMed":
        dataset = Planetoid(root="../data/PubMed", name="PubMed")
        data = dataset[0]
    elif args.dataset == "CoraFull":
        dataset = CoraFull(root="../data/CoraFull")
        data = dataset[0]
    elif args.dataset == "Amazon-Computers":
        dataset = Amazon(root="../data/Amazon-Computers", name="Computers")
        data = dataset[0]
    elif args.dataset == "Amazon-Photo":
        dataset = Amazon(root="../data/Amazon-Photo", name="Photo")
        data = dataset[0]
    elif args.dataset == "CitationFull-Cora":
        dataset = CitationFull(root="../data/CitationFull-Cora", name="Cora")
        data = dataset[0]
    elif args.dataset == "CitationFull-DBLP":
        dataset = CitationFull(root="../data/CitationFull-DBLP", name="DBLP")
        data = dataset[0]
    elif args.dataset == "Flickr":
        dataset = Flickr(root="../data/Flickr")
        data = dataset[0]
    elif args.dataset == "Reddit":
        dataset = Reddit(root="../data/Reddit")
        data = dataset[0]
    elif args.dataset == "Coauthor-CS":
        dataset = Coauthor(root="../data/Coauthor-CS", name="CS")
        data = dataset[0]
    elif args.dataset == "Coauthor-Physics":
        dataset = Coauthor(root="../data/Coauthor-Physics", name="Physics")
        data = dataset[0]

    print("Dataset loaded. ")

    if not hasattr(data, "train_mask"):
        data = dataset_split(data)

    print("val_mask:")
    print(hasattr(data, "val_mask"))

    assert hasattr(data, "val_mask") == True

    args.num_of_nodes = data.x.shape[0]

    enc = OneHotEncoder()
    enc.fit(data.y.view(-1).reshape(-1, 1))
    data.y = enc.transform(data.y.view(-1).reshape(-1, 1)).toarray()

    model = Node2Vec(
        data.edge_index,
        embedding_dim=args.embedding_dim,
        walk_length=20,
        context_size=10,
        rank_bound=args.rank,
        walks_per_node=10,
        num_negative_samples=1,
        p=args.p,
        q=args.q,
        sparse=True,
        num_nodes=args.num_of_nodes,
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)

    def train(epoch):
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device), epoch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def train_an_mlp(model_out, data, class_num):
        predictor = MlpProdDecoder(
            args.embedding_dim, hidden_size=256, class_num=class_num
        ).to(device)
        criterion = BCEWithLogitsLoss()

        optimizer = Adam(list(predictor.parameters()), lr=5e-3)

        train_labels = torch.from_numpy(data.y[data.train_mask]).to(device)

        for i in range(100):
            predictor.train()
            optimizer.zero_grad()

            edge_embeddings = model_out[data.train_mask]
            out = predictor(edge_embeddings)

            loss = criterion(out.to(device), train_labels.float())
            loss.backward()
            optimizer.step()

        predictor.eval()

        edge_embeddings = model_out
        out = predictor(edge_embeddings)

        return out

    def test(data, flag=0):
        model.eval()
        z = model(test=1)
        model_out = z.detach()

        out = train_an_mlp(model_out.detach(), data, class_num=data.y.shape[1])
        out_cpu = out.detach().cpu().numpy()

        out_cpu_val = np.argmax(out_cpu[data.val_mask], axis=1)
        out_cpu_test = np.argmax(out_cpu[data.test_mask], axis=1)

        out_cpu_val_multi_dim = softmax(out_cpu[data.val_mask], axis=1)
        out_cpu_test_multi_dim = softmax(out_cpu[data.test_mask], axis=1)

        val_labels = np.argmax(data.y[data.val_mask], axis=1)
        test_labels = np.argmax(data.y[data.test_mask], axis=1)

        roc_true_labels_val = np.unique(val_labels)
        roc_true_labels_test = np.unique(test_labels)

        roc_true_labels_val.sort()
        roc_true_labels_test.sort()

        roc_pred_labels_val = out_cpu_val_multi_dim[:, roc_true_labels_val]
        roc_pred_labels_test = out_cpu_test_multi_dim[:, roc_true_labels_test]

        # val
        accuracy_score_val = accuracy_score(val_labels, out_cpu_val)
        macro_f1_val = f1_score(val_labels, out_cpu_val, average="macro")
        micro_f1_val = f1_score(val_labels, out_cpu_val, average="micro")

        recall_macro_val = recall_score(val_labels, out_cpu_val, average="macro")
        recall_micro_val = recall_score(val_labels, out_cpu_val, average="micro")

        # test
        accuracy_score_test = accuracy_score(test_labels, out_cpu_test)
        macro_f1_test = f1_score(test_labels, out_cpu_test, average="macro")
        micro_f1_test = f1_score(test_labels, out_cpu_test, average="micro")

        recall_macro_test = recall_score(test_labels, out_cpu_test, average="macro")
        recall_micro_test = recall_score(test_labels, out_cpu_test, average="micro")

        rank = 0.000
        if flag == 1:
            if args.cluster == 1 and dataset != "Reddit" and dataset != "Flickr":
                rank = np.linalg.matrix_rank(model_out.detach().cpu().numpy())
            elif args.cluster == 0:
                rank = torch.linalg.matrix_rank(model_out.detach())

        return_dict = {}

        return_dict["accuracy_score_val"] = accuracy_score_val
        return_dict["macro_f1_val"] = macro_f1_val
        return_dict["micro_f1_val"] = micro_f1_val
        return_dict["recall_macro_val"] = recall_macro_val
        return_dict["recall_micro_val"] = recall_micro_val

        return_dict["accuracy_score_test"] = accuracy_score_test
        return_dict["macro_f1_test"] = macro_f1_test
        return_dict["micro_f1_test"] = micro_f1_test
        return_dict["recall_macro_test"] = recall_macro_test
        return_dict["recall_micro_test"] = recall_micro_test

        return_dict["rank"] = rank

        # degree analysis

        global_graph = build_graph_with_max_nodes(data.edge_index, args.num_of_nodes)

        the_concatenation = np.arange(args.num_of_nodes)[data.test_mask]

        degrees = np.array([global_graph.degree(node) for node in the_concatenation])

        degree_ranks = np.argsort(np.argsort(degrees))

        low_degree_nodes_marker = degree_ranks < (degree_ranks.shape[0] / 2.0)
        high_degree_nodes_marker = degree_ranks > (degree_ranks.shape[0] / 2.0)

        return_dict["accuracy_score_test_low_degree_nodes"] = accuracy_score(
            test_labels[low_degree_nodes_marker], out_cpu_test[low_degree_nodes_marker]
        )
        return_dict["macro_f1_test_low_degree_nodes"] = f1_score(
            test_labels[low_degree_nodes_marker],
            out_cpu_test[low_degree_nodes_marker],
            average="macro",
        )
        return_dict["micro_f1_test_low_degree_nodes"] = f1_score(
            test_labels[low_degree_nodes_marker],
            out_cpu_test[low_degree_nodes_marker],
            average="micro",
        )

        return_dict["recall_macro_test_low_degree_nodes"] = recall_score(
            test_labels[low_degree_nodes_marker],
            out_cpu_test[low_degree_nodes_marker],
            average="macro",
        )
        return_dict["recall_micro_test_low_degree_nodes"] = recall_score(
            test_labels[low_degree_nodes_marker],
            out_cpu_test[low_degree_nodes_marker],
            average="micro",
        )

        return_dict["accuracy_score_test_high_degree_nodes"] = accuracy_score(
            test_labels[high_degree_nodes_marker],
            out_cpu_test[high_degree_nodes_marker],
        )
        return_dict["macro_f1_test_high_degree_nodes"] = f1_score(
            test_labels[high_degree_nodes_marker],
            out_cpu_test[high_degree_nodes_marker],
            average="macro",
        )
        return_dict["micro_f1_test_high_degree_nodes"] = f1_score(
            test_labels[high_degree_nodes_marker],
            out_cpu_test[high_degree_nodes_marker],
            average="micro",
        )

        return_dict["recall_macro_test_high_degree_nodes"] = recall_score(
            test_labels[high_degree_nodes_marker],
            out_cpu_test[high_degree_nodes_marker],
            average="macro",
        )
        return_dict["recall_micro_test_high_degree_nodes"] = recall_score(
            test_labels[high_degree_nodes_marker],
            out_cpu_test[high_degree_nodes_marker],
            average="micro",
        )

        return return_dict, out_cpu[data.test_mask]

    for epoch in tqdm(range(0, args.epochs)):
        loss = train(epoch)

        if epoch == args.epoch_threshold:
            model.V.weight += 1e-4 * torch.randn(args.rank, args.embedding_dim).to(
                device
            )
            model.V.weight.requires_grad = True
            model.U.weight.requires_grad = False
            for g in optimizer.param_groups:
                g["lr"] = 0.001

    return_dict, preds_all_test = test(data, flag=1)
    print(return_dict)

    comp = list(return_dict.values())

    formatted_comp = [f"{num:.4f}" for num in comp]

    comp_string = ",".join(formatted_comp)

    prediction_dict = {
        "predicted": preds_all_test,
        "num_of_positives": data.y[data.test_mask],
        "test_mask": np.arange(args.num_of_nodes)[data.test_mask],
    }

    if args.write == 1:
        if args.cluster == 1:
            assert 1 == 0
            # if not os.path.exists("../../1_results/1_node_classification_completed.txt"):
            #     print(f"1_node_classification_completed.txt does not exist, creating it...")

            #     with open("../../1_results/1_node_classification_completed.txt", "a") as f:
            #         formatted_comp_keys = list(return_dict.keys())
            #         comp_string_keys = ",".join(formatted_comp_keys)
            #         f.write("model,dataset,rank_bound,embedding_dim,p,q,feature_dim,seed," + comp_string_keys + "\n")

            # with open(f"../../1_results/1_node_classification_completed.txt", "a") as file:
            #     line = f"deepwalk,{args.dataset},{args.rank:.1f},{args.embedding_dim:.1f},{args.p:.1f},{args.q:.1f},{args.feature_dim:.8f},{args.seed:.1f}," + comp_string + "\n"
            #     file.write(line)

            # np.save(f"../../1_classification/deepwalk_{args.dataset}_{args.rank:.0f}_{args.embedding_dim:.0f}_{args.p:.0f}_{args.q:.0f}_{args.feature_dim:.8f}_{args.seed:.0f}.npy", prediction_dict)

        elif args.cluster == 0:
            if not os.path.exists("../example_results/2_node_classification_completed.txt"):
                print(
                    f"2_node_classification_completed.txt does not exist, creating it..."
                )

                with open("../example_results/2_node_classification_completed.txt", "a") as f:
                    formatted_comp_keys = list(return_dict.keys())
                    comp_string_keys = ",".join(formatted_comp_keys)
                    f.write(
                        "model,dataset,rank_bound,embedding_dim,p,q,feature_dim,seed,"
                        + comp_string_keys
                        + "\n"
                    )

            with open(f"../example_results/2_node_classification_completed.txt", "a") as file:
                line = (
                    f"deepwalk_prop,{args.dataset},{args.rank:.1f},{args.embedding_dim:.1f},{args.p:.1f},{args.q:.1f},{args.feature_dim:.8f},{args.seed:.1f},"
                    + comp_string
                    + "\n"
                )
                file.write(line)

            np.save(
                f"../example_results/deepwalk_prop_nc_{args.dataset}_{args.rank:.0f}_{args.embedding_dim:.0f}_{args.p:.0f}_{args.q:.0f}_{args.feature_dim:.8f}_{args.seed:.0f}.npy",
                prediction_dict,
            )
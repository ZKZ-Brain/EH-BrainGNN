import torch
from torch_geometric.nn import MessagePassing
from math import sqrt
from utils import calculate_MI
import torch.nn.functional as F

criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.BCELoss()

class SubgraphGenerator(torch.nn.Module):
 
    def __init__(self, model, device):
        super(SubgraphGenerator, self).__init__()

        self.model = model
        self.device = device
        self.reg_coefs = (0.0001, 0.0001)


    def clear_masks(self,model):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None


    def set_masks(self, model, edgemask):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = edgemask

    def set_edgemasks(self, x, edge_index):
        (N, F), E = x.size(), edge_index.size(1)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E)*std)

        return self.edge_mask

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make samplign less deterministic
        :param training: If set to false, the samplign will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + 0.0001 # If bias is 0, we run into problems
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()) + (1-bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs= gate_inputs.to(self.device)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)

        return graph

    def _loss(self, masked_pred, original_pred, edge_mask, reg_coefs,pre_graphemb,graphemb):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]
        EPS = 1e-15

        # Regularization losses
        mask = torch.sigmoid(edge_mask)
        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)*0.1

        # Explanation loss
        cce_loss = criterion(masked_pred, original_pred)
        mutual_loss = entropy_reg * calculate_MI(pre_graphemb,graphemb)

        return cce_loss + size_loss + mask_ent_loss + mutual_loss

    def forward(self, data):
        x, edge_index,batch= data.x.to(self.device), data.edge_index.to(self.device),data.batch.to(self.device)
        self.clear_masks(self.model)
        pre_logits, pre_graphemb, _ = self.model(x.to(self.device),edge_index.to(self.device),batch.to(self.device))
        edge_mask = self.set_edgemasks(x, edge_index).to(self.device)
        edge_mask = self._sample_graph(edge_mask)
        # edge_mask = torch.sigmoid(edge_mask).to(self.device)
        # edge_mask = F.gumbel_softmax(edge_mask)
        self.set_masks(self.model, edge_mask)
        logits,graphemb,node_emb = self.model(x.to(self.device),edge_index.to(self.device),batch.to(self.device))
        labels = torch.LongTensor(data.y).to(self.device)
        loss = self._loss(logits,labels,edge_mask, self.reg_coefs,pre_graphemb,graphemb)

        return logits, loss

# gcn_base.py
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SGConv

class SGC(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(SGC, self).__init__()
        self.conv1 = SGConv(num_features, num_classes, K=2, cached=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)

class GCNClassifier:
    def __init__(self, gcn_type, rks, pN, number_neighbors=40):
        self.pK = number_neighbors
        self.pN = pN
        self.rks = rks
        self.pLR = 0.001
        self.pNNeurons = 32
        self.pNEpochs = 50
        self.gcn_type = gcn_type

    def prepare(self, test_index, train_index, features, labels, analy=False):
        print('Creating Masks...')
        self.train_mask = [False] * self.pN
        self.val_mask = [False] * self.pN
        self.test_mask = [False] * self.pN

        for index in train_index:
            self.train_mask[index] = True
        for index in test_index:
            self.test_mask[index] = True

        self.train_mask = torch.tensor(self.train_mask)
        self.val_mask = torch.tensor(self.val_mask)
        self.test_mask = torch.tensor(self.test_mask)

        print('Set Labels...')
        y = labels
        self.numbersOfClasses = max(y) + 1
        self.y = torch.tensor(y)

        self.x = torch.tensor(features)
        self.pNFeatures = len(features[0])

        if analy == False:
            self.create_graph('jaccard')

    def compute_RBO(self, rks, top_k, limiar):
        edge_index= []
        r= 0.9
        for z in range(len(rks)):
            for p in range(len(rks)):
                stored= set()
                acum_inter= 0
                score= 0
                img1_leftover= set()
                img2_leftover= set()

                for i in range(top_k):
                    img1_elm= rks[z][i]
                    img2_elm= rks[p][i]

                    if img1_elm not in stored and img1_elm == img2_elm:
                        acum_inter += 1
                        stored.add(img1_elm)
                    else:
                        if img1_elm not in stored:
                            if img1_elm in img2_leftover:
                                acum_inter += 1
                                stored.add(img1_elm)
                                img2_leftover.remove(img1_elm)
                            else:
                                img1_leftover.add(img1_elm)
                        if img2_elm not in stored:
                            if img2_elm in img1_leftover:
                                acum_inter += 1
                                stored.add(img2_elm)
                                img1_leftover.remove(img2_elm)
                            else:
                                img2_leftover.add(img2_elm)
                    score += (r**((i+1)-1)) * (acum_inter / (i+1))
                scrN= (1-r) * score
                
                if scrN >= limiar:
                    edge_index.append([z,p])

        edge_index= torch.tensor(edge_index)
        self.edge_index= edge_index.t().contiguous()

    def compute_jaccard(self, rks, top_k, limiar):
        edge_index = []
        for i in range(len(rks)):
            for j in range(len(rks)):
                b = len(set(rks[i][:top_k]) & set(rks[j][:top_k])) / len(set(rks[i][:top_k]) | set(rks[j][:top_k]))
                if b >= limiar:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index)
        self.edge_index = edge_index.t().contiguous()
    
    def create_graph(self, correlation_measure, limiar=0.6):
        if correlation_measure == 'normal':
            print('Making edge list...')
            self.top_k = self.pK
            edge_index = []
            for img1 in range(len(self.rks)):
                for pos in range(self.top_k):
                    img2 = self.rks[img1][pos]
                    edge_index.append([img1, img2])
            edge_index = torch.tensor(edge_index)
            self.edge_index = edge_index.t().contiguous()
        if correlation_measure == 'jaccard':
            self.compute_jaccard(self.rks, self.pK, limiar)
        if correlation_measure == 'RBO':
            self.compute_RBO(self.rks, self.pK, limiar)

    def train_and_predict(self):
        print('Loading data object...')
        data = Data(x=self.x.float(), edge_index=self.edge_index, y=self.y, 
                    test_mask=self.test_mask, train_mask=self.train_mask, val_mask=self.val_mask)
        model = SGC(self.pNFeatures, self.numbersOfClasses)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.pLR, weight_decay=5e-4)

        print('Training')   
        model.train()
        for epoch in range(self.pNEpochs):
            #print(f'Training epoch: {epoch}')
            optimizer.zero_grad()
            out = model(data)
            data.y = torch.tensor(data.y, dtype=torch.long)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        print('Evaluating...')
        model.eval()
        _, pred = model(data).max(dim=1)
        pred = torch.masked_select(pred, data.test_mask)
        embeddings = model(data)
        return embeddings, pred.tolist()

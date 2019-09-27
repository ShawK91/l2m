class KNet(nn.Module):
    """Actor model

        Parameters:
              args (object): Parameter class
    """

    def __init__(self, input_dim, neuron_dim, col_dim, output_dim):
        super(KNet, self).__init__()

        #Core Network
        self.linear1 = nn.Linear(input_dim, neuron_dim)
        self.linear2 = nn.Linear(neuron_dim, neuron_dim)
        self.linear_out = nn.Linear(neuron_dim, output_dim)


        #Column Distribution
        self.map1, self.entropy1 = self.init_allocation(neuron_dim, col_dim)
        self.map2, self.entropy2 = self.init_allocation(neuron_dim, col_dim)

        self.apply(weights_init_)


    def forward(self, input):
        """Method to forward propagate through the actor's graph

            Parameters:
                  input (tensor): [batch_size, entry]

            Returns:
                  action (tensor): [batch_size, entry]


        """
        #First Layer
        out = self.linear1(input)
        out = self.apply_map(out, self.map1, self.entropy1)

        #Second Layer
        out = self.linear2(out)
        out = self.apply_map(out, self.map2, self.entropy2)

        #Final Layer
        out = self.linear_out(out)

        return out

    def get_norm_stats(self):
        minimum = min([torch.min(param).item() for param in self.parameters()])
        maximum = max([torch.max(param).item() for param in self.parameters()])
        means = [torch.mean(torch.abs(param)).item() for param in self.parameters()]
        mean = sum(means)/len(means)

        return minimum, maximum, mean

    def apply_map(self, input, map, entropy):

        ls = []
        for i, col in enumerate(map):
            local_out = torch.softmax(input[:,col], dim=1)
            #nput[:,col] = local_out
            ls.append(local_out)

            #Entropy
            entropy[i] = (local_out * torch.log(local_out)).sum()

        out = torch.cat(ls, axis=1)
        return out


    def map_update(self):
        if random.random<0.05:
            max_ent_col = self.entropy1.index(max([v.item() for v in self.entropy1]))
            min_ent_col = self.entropy1.index(min([v.item() for v in self.entropy1]))
            if self.map1[max_ent_col] > 2:
                node = self.map1[max_ent_col].pop(0)
                self.map1[min_ent_col].append(node)

        if random.random<0.05:
            max_ent_col = self.entropy2.index(max([v.item() for v in self.entropy2]))
            min_ent_col = self.entropy2.index(min([v.item() for v in self.entropy2]))
            if self.map2[max_ent_col] > 2:
                node = self.map2[max_ent_col].pop(0)
                self.map2[min_ent_col].append(node)

    def init_allocation(self, neurone_dim, col_dim):
        allocation = []
        ration = int(neurone_dim/col_dim)
        counter = 0
        for i in range(col_dim):
            allocation.append([])
            for _ in range(ration):
                allocation[-1].append(counter)
                counter+=1

        #Put all remaining to last entry
        while counter < neurone_dim:
            allocation[-1].append(counter)
            counter += 1

        entropy = [None for _ in range(col_dim)]

        return allocation, entropy

# class EEGDataset(Dataset):
#     def __init__(self, path):
#         self.files = list(Path(path).rglob("*.csv"))
#         self.mapping = {"error": 0, "correct": 1}

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         if not isinstance(idx, slice):
#             return self._get_one_item(self.files[idx])
#         else:
#             data, labels = [], []
#             for file in self.files[idx]:
#                 one_data, label = self._get_one_item(file)
#                 data.append(one_data)
#                 labels.append(label)
#             return torch.stack(data), labels

#     def _get_one_item(self, file):
#         label = self.mapping[file.parent.name]
#         data = pd.read_csv(file).to_numpy()[:3000].transpose().astype(np.float32)
#         return torch.tensor(data), label

    # def get_weighted_sampler(self):
    #     labels = [self.mapping[file.parent.name] for file in self.files]
    #     counts = np.unique(labels, return_counts=True)[1]
    #     weights_classes = 1 / (counts / sum(counts))
    #     weights_classes /= sum(weights_classes)
    #     weights = [weights_classes[x] for x in labels]
    #     sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    #     return sampler


# train_dataset = EEGDataset("/home/tomek/nauka/mne/data/train")
# test_dataset = EEGDataset("/home/tomek/nauka/mne/data/test")

# sampler = train_dataset.get_weighted_sampler()
# train_loader = DataLoader(train_dataset, batch_size=10, sampler=sampler)
# test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.kernels = [50, 100, 150, 300, 500, 700, 900]
#         self.cnn_size = 184
#         self.cnn = nn.Sequential(
#             nn.Conv1d(21, self.kernels[0], 3, stride=1),
#             nn.Conv1d(self.kernels[0], self.kernels[1], 3, stride=1),
#             nn.Conv1d(self.kernels[1], self.kernels[2], 3, stride=1),
#             nn.Conv1d(self.kernels[2], self.kernels[3], 5, stride=2),
#             nn.Conv1d(self.kernels[3], self.kernels[4], 5, stride=2),
#             nn.Conv1d(self.kernels[4], self.kernels[5], 5, stride=2),
#             nn.Conv1d(self.kernels[5], self.kernels[6], 5, stride=2),
#         )
#         # data, _ = train_dataset[0]
#         # print(self.cnn(data.unsqueeze(0)).shape)
#         self.classifier = nn.Sequential(
#             nn.Linear(self.cnn_size * self.kernels[-1], 1), nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.cnn(x)
#         return self.classifier(x.view(-1, self.cnn_size * self.kernels[-1]))


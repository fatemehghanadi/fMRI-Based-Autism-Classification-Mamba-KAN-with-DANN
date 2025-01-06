# Define PyTorch Dataset
class fMRIDataset_domain(Dataset):
    def __init__(self, fc, cc, labels, age, gender,numeric_institutions, fiq, viq, piq, eye,numeric_handedness ):
        self.fc = torch.tensor(fc, dtype=torch.float32)
        self.cc = torch.tensor(cc, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.age = torch.tensor(age, dtype=torch.float32)
        self.gender = torch.tensor(gender, dtype=torch.float32)
        self.handedness = torch.tensor(numeric_handedness, dtype=torch.float32)
        self.fiq = torch.tensor(fiq, dtype=torch.float32)
        self.viq = torch.tensor(viq, dtype=torch.float32)
        self.piq = torch.tensor(piq, dtype=torch.float32)
        self.eye = torch.tensor(eye, dtype=torch.float32)
        self.site = torch.tensor(numeric_institutions, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fc_tensor = torch.tensor(self.fc, dtype=torch.float32).to(device)
        cc_tensor = torch.tensor(self.cc, dtype=torch.float32).to(device)


        # combined_tensors = CombinedTensors(fc_tensor[idx], cc_tensor[idx])
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # combined_tensors = combined_tensors.to(device)



        return [fc_tensor[idx], cc_tensor[idx]], self.labels[idx], self.site[idx]

def custom_collate_domain(batch):
    """
    Custom collate function to process a batch of (fc, cc, labels).

    Args:
        batch (list of tuples): Each tuple contains (fc, cc, labels).
            - fc: torch.Tensor, shape (m, n)
            - cc: torch.Tensor, shape (p)
            - labels: torch.Tensor, shape (1) or (num_classes)

    Returns:
        concatenated_features (torch.Tensor): Batched features of shape (batch_size, m*n + p).
        labels (torch.Tensor): Batched labels.
    """
    # Flatten and concatenate fc and cc for each sample in the batch
    features = [torch.cat((item[0][0].flatten(), item[0][1]), dim=0) for item in batch]
    # Stack all features to form a batch
    concatenated_features = torch.stack(features)
    # Extract and stack labels
    labels = torch.stack([item[1] for item in batch])
    domains = torch.stack([item[2] for item in batch])


    return concatenated_features, labels, domains
# Define PyTorch Dataset
class fMRIDataset(Dataset):
    def __init__(self, fc, cc, labels, age, gender,numeric_institutions, fiq, viq, piq, eye,numeric_handedness ):
        self.fc = torch.tensor(fc, dtype=torch.float32)
        self.cc = torch.tensor(cc, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.age = torch.tensor(age, dtype=torch.float32)
        self.gender = torch.tensor(gender, dtype=torch.float32)
        self.handedness = torch.tensor(numeric_handedness, dtype=torch.float32)
        self.fiq = torch.tensor(fiq, dtype=torch.float32)
        self.viq = torch.tensor(viq, dtype=torch.float32)
        self.piq = torch.tensor(piq, dtype=torch.float32)
        self.eye = torch.tensor(eye, dtype=torch.float32)
        self.site = torch.tensor(numeric_institutions, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fc_tensor = torch.tensor(self.fc, dtype=torch.float32).to(device)
        cc_tensor = torch.tensor(self.cc, dtype=torch.float32).to(device)


        # combined_tensors = CombinedTensors(fc_tensor[idx], cc_tensor[idx])
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # combined_tensors = combined_tensors.to(device)



        return [fc_tensor[idx], cc_tensor[idx]] , self.labels[idx]

# Define PyTorch Dataset
class fMRIDataset_target(Dataset):
    def __init__(self, fc, cc, labels, age, gender,numeric_institutions, fiq, viq, piq, eye,numeric_handedness ):
        self.fc = torch.tensor(fc, dtype=torch.float32)
        self.cc = torch.tensor(cc, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.age = torch.tensor(age, dtype=torch.float32)
        self.gender = torch.tensor(gender, dtype=torch.float32)
        self.handedness = torch.tensor(numeric_handedness, dtype=torch.float32)
        self.fiq = torch.tensor(fiq, dtype=torch.float32)
        self.viq = torch.tensor(viq, dtype=torch.float32)
        self.piq = torch.tensor(piq, dtype=torch.float32)
        self.eye = torch.tensor(eye, dtype=torch.float32)
        self.site = torch.tensor(numeric_institutions, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fc_tensor = torch.tensor(self.fc, dtype=torch.float32).to(device)
        cc_tensor = torch.tensor(self.cc, dtype=torch.float32).to(device)


        # combined_tensors = CombinedTensors(fc_tensor[idx], cc_tensor[idx])
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # combined_tensors = combined_tensors.to(device)



        return [fc_tensor[idx], cc_tensor[idx]], self.site[idx]

def custom_collate(batch):
    """
    Custom collate function to process a batch of (fc, cc, labels).

    Args:
        batch (list of tuples): Each tuple contains (fc, cc, labels).
            - fc: torch.Tensor, shape (m, n)
            - cc: torch.Tensor, shape (p)
            - labels: torch.Tensor, shape (1) or (num_classes)

    Returns:
        concatenated_features (torch.Tensor): Batched features of shape (batch_size, m*n + p).
        labels (torch.Tensor): Batched labels.
    """
    # Flatten and concatenate fc and cc for each sample in the batch
    features = [torch.cat((item[0][0].flatten(), item[0][1]), dim=0) for item in batch]
    # Stack all features to form a batch
    concatenated_features = torch.stack(features)
    # Extract and stack labels
    labels = torch.stack([item[1] for item in batch])

    return concatenated_features, labels

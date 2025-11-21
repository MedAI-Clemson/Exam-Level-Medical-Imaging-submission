import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import default_collate

def pad(h, num_imgs):
    # reshape as batch of studies and pad
    start_ix = 0
    h_ls = []
    for n in num_imgs:
        h_ls.append(h[start_ix:(start_ix+n)])
        start_ix += n
    h = pad_sequence(h_ls)

    return h

def paddp(h, num_imgs, num_series):
    # reshape as batch of studies and pad
    start_ix = 0
    h_ls = []
    for n in num_imgs:
        h_ls.append(h[start_ix:(start_ix+n)])
        start_ix += n
    h = pad_sequence(h_ls)

    return h

def num_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def custom_collate(batch):
    batch_studies = torch.concat([record["study"] for record in batch])
    batch_lengths = [len(record["study"]) for record in batch]
    
    record = {
        "study": batch_studies,
        "num_imgs": batch_lengths
    }
    for b in batch:
        b.pop('study')
    record.update(default_collate(batch))
    
    return record


def collate_fn(batch):
    # print("collate_fn called")
    # Initialize empty lists for combined bags and labels
    combined_bag_images = []
    combined_instance_labels = []
    combined_bag_labels = []  # Collect all bag labels
    combined_patient_ids = []
    markers = [0]  # Start with 0, the index for the start of the first bag
    
    total_images = 0
    for bag, (bag_label, instance_labels, patient_id) in batch:
        num_images = bag.shape[0]  # Number of images in the current bag
        total_images += num_images
        combined_bag_images.append(bag)
        combined_instance_labels.append(instance_labels)
        combined_bag_labels.append(bag_label)  # Collect bag labels for each study bag
        combined_patient_ids.append(patient_id)
        markers.append(total_images)  # Append cumulative image count

    # Combine all study bags into one along the 0th dimension (number of images)
    combined_bag_images = torch.cat(combined_bag_images, dim=0)  # Combined shape: [n1 + n2 + ..., channel, h, w]
    combined_instance_labels = torch.cat(combined_instance_labels, dim=0)  # Combined instance labels

    return combined_bag_images, (combined_bag_labels, combined_instance_labels, combined_patient_ids, markers)
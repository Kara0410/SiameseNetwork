import torch

from ai_services.siamese.losses import ContrastiveLoss, TripletLoss


def test_contrastive_loss_is_zero_for_identical_similar_pairs():
    loss_fn = ContrastiveLoss(margin=1.0)
    a = torch.zeros(4, 8)
    b = torch.zeros(4, 8)
    label = torch.zeros(4)  # 0 = similar pair

    # `F.pairwise_distance` adds a tiny epsilon before squaring, so this is ~0 but
    # not exactly representable as 0.0.
    assert loss_fn(a, b, label).item() < 1e-9


def test_contrastive_loss_penalizes_close_dissimilar_pairs():
    loss_fn = ContrastiveLoss(margin=1.0)
    a = torch.zeros(1, 8)
    b = torch.zeros(1, 8)
    label = torch.ones(1)  # 1 = dissimilar pair, but embeddings are identical

    loss = loss_fn(a, b, label)

    assert loss.item() > 0.0


def test_triplet_loss_is_zero_when_well_separated():
    loss_fn = TripletLoss(margin=1.0)
    anchor = torch.zeros(2, 8)
    positive = torch.full((2, 8), 0.01)
    negative = torch.full((2, 8), 5.0)

    assert loss_fn(anchor, positive, negative).item() == 0.0


def test_triplet_loss_is_positive_when_negative_is_closer():
    loss_fn = TripletLoss(margin=1.0)
    anchor = torch.zeros(2, 8)
    positive = torch.full((2, 8), 5.0)
    negative = torch.full((2, 8), 0.01)

    assert loss_fn(anchor, positive, negative).item() > 0.0

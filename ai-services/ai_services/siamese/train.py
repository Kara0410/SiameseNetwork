"""CLI for fine-tuning `SiameseEmbeddingNet` with triplet loss.

Modernized from `legacy/SN-TripletLoss/trainModelTriplet.py`: configurable via
arguments instead of module constants, uses the shared device/config helpers, and
saves a checkpoint that the `siamese` embedding backend can load via the
`VISIONIQ_SIAMESE_CHECKPOINT` environment variable.

Example::

    python -m ai_services.siamese.train \\
        --anchor data/anchor --positive data/positive --negative data/negative \\
        --epochs 5 --out models/siamese.pth
"""

import argparse
import os

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

from .. import config
from .dataset import TripletFaceDataset
from .losses import TripletLoss
from .network import SiameseEmbeddingNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True, help="Directory of anchor images")
    parser.add_argument("--positive", required=True, help="Directory of positive images")
    parser.add_argument("--negative", required=True, help="Directory of negative images")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--out", default="models/siamese.pth", help="Checkpoint output path")
    args = parser.parse_args()

    dataset = TripletFaceDataset(args.anchor, args.positive, args.negative)
    val_size = max(1, int(0.2 * len(dataset)))
    train_data, val_data = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    device = config.DEVICE
    model = SiameseEmbeddingNet(embedding_dim=args.embedding_dim).to(device)
    loss_fn = TripletLoss(margin=args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_loss = _run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        model.eval()
        val_loss = _run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)
        print(f"epoch {epoch + 1}/{args.epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved checkpoint to {args.out}")


def _run_epoch(model, loader, loss_fn, optimizer, device, train: bool) -> float:
    total_loss, batches = 0.0, 0
    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for anchor, positive, negative in loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            anchor_embedding = model.forward_once(anchor)
            positive_embedding = model.forward_once(positive)
            negative_embedding = model.forward_once(negative)
            loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            batches += 1

    return total_loss / max(batches, 1)


if __name__ == "__main__":
    main()

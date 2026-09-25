"""Memory-mapped token data for stage 1 next-token prediction."""

from __future__ import annotations

from pathlib import Path

import torch


class TokenDataset:
    """Read a uint16 token stream and sample contiguous training windows."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self.path.stat().st_size % 2:
            raise ValueError(f"Token file has an odd byte length: {self.path}")
        self.data = torch.from_file(
            str(self.path),
            shared=False,
            size=self.path.stat().st_size // 2,
            dtype=torch.uint16,
        )

    def __len__(self) -> int:
        return self.data.numel()

    def get_batch(
        self,
        batch_size: int,
        block_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if len(self) <= block_size:
            raise ValueError(
                f"Dataset {self.path} has {len(self)} tokens; "
                f"at least {block_size + 1} are required"
            )

        starts = torch.randint(
            0,
            len(self) - block_size,
            (batch_size,),
            generator=generator,
            device="cpu",
        )
        offsets = torch.arange(block_size, dtype=torch.long)
        indices = starts[:, None] + offsets[None, :]
        inputs = self.data[indices]
        targets = self.data[indices + 1]
        return (
            inputs.to(device=device, dtype=torch.long),
            targets.to(device=device, dtype=torch.long),
        )


def load_data(data_dir: str | Path) -> tuple[TokenDataset, TokenDataset]:
    data_dir = Path(data_dir)
    return TokenDataset(data_dir / "train.bin"), TokenDataset(data_dir / "val.bin")

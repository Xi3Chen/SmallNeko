"""PyTorch batch loader for stage 0 uint16 token streams."""

from __future__ import annotations

from pathlib import Path

import torch


class TokenDataset:
    """Memory-map a token stream and sample next-token prediction batches."""

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
        )
        inputs = torch.stack([self.data[start : start + block_size] for start in starts.tolist()])
        targets = torch.stack([self.data[start + 1 : start + block_size + 1] for start in starts.tolist()])
        return inputs.to(device=device, dtype=torch.long), targets.to(device=device, dtype=torch.long)


def load_stage0_data(data_dir: str | Path) -> tuple[TokenDataset, TokenDataset]:
    data_dir = Path(data_dir)
    return TokenDataset(data_dir / "train.bin"), TokenDataset(data_dir / "val.bin")

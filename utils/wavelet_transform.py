from typing import Dict, Literal, NamedTuple
import torch
import ptwt

DETAIL_NAMES = ["horizontal", "vertical", "diagonal"]


class WaveletCoeffs(NamedTuple):
    """
    Structured wavelet coefficients container
    (based on NamedTuple for type hinting and documentation).
    """

    A: torch.Tensor
    # D1, D2, ...: torch.Tensor，shape: (B, C, 3, H, W)


WaveletCoeffDict = Dict[str, torch.Tensor]


class WaveletTransform:
    def __init__(
        self,
        wavelet: str,
        level: int,
        mode: Literal["constant", "reflect", "zero", "periodic"] = "reflect",
    ) -> None:
        if level < 1:
            raise ValueError("Level must be >= 1")
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

    def transform(self, input_tensor: torch.Tensor) -> WaveletCoeffDict:
        """
        Execute 2D wavelet decomposition and return structured coefficient dictionary.

        :param:
            input_tensor: torch.Tensor，shape: (B, C, H, W)

        :return:
            WaveletCoeffDict：
                - "A": approximation coefficient
                - "D1", "D2", ..., "D{level}": detail coefficients
                  each D{l}'s shape: (B, C, 3, H_l, W_l)
                  on dimension 3: [horizontal, vertical, diagonal]
        """
        coeffs = ptwt.wavedec2(
            input_tensor,
            wavelet=self.wavelet,
            level=self.level,
            mode=self.mode,
        )

        # coeffs[0]: approximation
        # coeffs[1:]: detail tuples per level
        result: WaveletCoeffDict = {"A": coeffs[0]}

        for l in range(1, self.level + 1):
            level_details = coeffs[l]  # Tuple[Tensor, Tensor, Tensor]
            if len(level_details) != 3:
                raise RuntimeError(
                    f"Expected 3 detail coefficients at level {l}, got {len(level_details)}"
                )
            # concat as (B, C, 3, H, W)
            stacked = torch.stack(level_details, dim=2)
            result[f"D{l}"] = stacked

        return result

    def reverse(self, input: WaveletCoeffDict) -> torch.Tensor:
        """
        Execute inverse 2D wavelet reconstruction.

        :param:
            input: WaveletCoeffDict - "A" and "D1", "D2", ..., "D{level}"

        :return:
            torch.Tensor - shape: (B, C, H, W)
        """

        coeffs_list = [input["A"]]
        

        for l in range(1, self.level + 1):
            d_tensor = input[f"D{l}"]  # (B, C, 3, H, W)
            if d_tensor.shape[2] != 3:
                raise ValueError(f"D{l} must have 3 detail directions, got shape {d_tensor.shape}")

            # horizontal, vertical, diagonal
            H, V, D = torch.unbind(d_tensor, dim=2)  # each is (B, C, H, W)
            coeffs_list.append((H, V, D))

        reconstructed = ptwt.waverec2(coeffs_list, wavelet=self.wavelet)

        return reconstructed.clone().contiguous()

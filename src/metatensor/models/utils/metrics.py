from typing import Dict, List, Tuple

from metatensor.torch import TensorMap


class RMSEAccumulator:
    """Accumulates the RMSE between predictions and targets for an arbitrary
    number of keys, each corresponding to one target."""

    def __init__(self):
        """Initialize the accumulator."""
        self.information: Dict[str, Tuple[float, int]] = {}

    def update(self, predictions: Dict[str, TensorMap], targets: Dict[str, TensorMap]):
        """Updates the accumulator with new predictions and targets.

        :param predictions: A dictionary of predictions, where the keys correspond
            to the keys in the targets dictionary, and the values are the predictions.

        :param targets: A dictionary of targets, where the keys correspond to the keys
            in the predictions dictionary, and the values are the targets.
        """

        for key, target in targets.items():
            if key not in self.information:
                self.information[key] = (0.0, 0)
            prediction = predictions[key]

            self.information[key] = (
                self.information[key][0]
                + ((prediction.block().values - target.block().values) ** 2)
                .sum()
                .item(),
                self.information[key][1] + prediction.block().values.numel(),
            )

            for gradient_name, target_gradient in target.block().gradients():
                if f"{key}_{gradient_name}_gradients" not in self.information:
                    self.information[f"{key}_{gradient_name}_gradients"] = (0.0, 0)
                prediction_gradient = prediction.block().gradient(gradient_name)
                self.information[f"{key}_{gradient_name}_gradients"] = (
                    self.information[f"{key}_{gradient_name}_gradients"][0]
                    + ((prediction_gradient.values - target_gradient.values) ** 2)
                    .sum()
                    .item(),
                    self.information[f"{key}_{gradient_name}_gradients"][1]
                    + prediction_gradient.values.numel(),
                )

    def finalize(self, not_per_atom: List[str]) -> Dict[str, float]:
        """Finalizes the accumulator and return the RMSE for each key.

        All keys will be returned as "{key} RMSE (per atom)" in the output dictionary,
        unless ``key`` contains one or more of the strings in ``not_per_atom``,
        in which case "{key} RMSE" will be returned.
        """

        finalized_info = {}
        for key, value in self.information.items():
            if any([s in key for s in not_per_atom]):
                out_key = f"{key} RMSE"
            else:
                out_key = f"{key} RMSE (per atom)"
            finalized_info[out_key] = (value[0] / value[1]) ** 0.5

        return finalized_info


class MAEAccumulator:
    """Accumulates the MAE between predictions and targets for an arbitrary
    number of keys, each corresponding to one target."""

    def __init__(self):
        """Initialize the accumulator."""
        self.information: Dict[str, Tuple[float, int]] = {}

    def update(self, predictions: Dict[str, TensorMap], targets: Dict[str, TensorMap]):
        """Updates the accumulator with new predictions and targets.

        :param predictions: A dictionary of predictions, where the keys correspond
            to the keys in the targets dictionary, and the values are the predictions.

        :param targets: A dictionary of targets, where the keys correspond to the keys
            in the predictions dictionary, and the values are the targets.
        """

        for key, target in targets.items():
            if key not in self.information:
                self.information[key] = (0.0, 0)
            prediction = predictions[key]

            self.information[key] = (
                self.information[key][0]
                + (prediction.block().values - target.block().values)
                .abs()
                .sum()
                .item(),
                self.information[key][1] + prediction.block().values.numel(),
            )

            for gradient_name, target_gradient in target.block().gradients():
                if f"{key}_{gradient_name}_gradients" not in self.information:
                    self.information[f"{key}_{gradient_name}_gradients"] = (0.0, 0)
                prediction_gradient = prediction.block().gradient(gradient_name)
                self.information[f"{key}_{gradient_name}_gradients"] = (
                    self.information[f"{key}_{gradient_name}_gradients"][0]
                    + (prediction_gradient.values - target_gradient.values)
                    .abs()
                    .sum()
                    .item(),
                    self.information[f"{key}_{gradient_name}_gradients"][1]
                    + prediction_gradient.values.numel(),
                )

    def finalize(self) -> Dict[str, float]:
        """Finalizes the accumulator and return the MAE for each key."""

        finalized_info = {}
        for key, value in self.information.items():
            finalized_info[f"{key} MAE"] = value[0] / value[1]

        return finalized_info

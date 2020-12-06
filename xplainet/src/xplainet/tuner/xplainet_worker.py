from xplainet.tuner.abstract_worker import AbstractWorker

# from xgboost import XGBClassifier
# import torch
from copy import deepcopy
from xplainet.model import build_model


# params,
# lconv_dim=[4],
# lconv_num_dim=[8],
# emb_size=16,
# # For this problem, we need "tanh" as first layer, or else to standard scale the data beforehand
# activation_num_first_layer=None,  # "tanh",
# output_activation=None,
# output_dim=1,  # np.unique(y_train).shape[0],
# return super().get_model_params_keys()  # + ["tree_method"]

DEFAULT_GRID = {
    # "params": [],
    # "lconv_dim": [[4], [8], [16], [32], [32, 8], [16, 4]],
    # "lconv_num_dim": [[4], [8], [16], [32], [32, 8], [16, 4]],
    "emb_size": {"type": "int", "lower": 1, "upper": 128, "default_value": 16},
    # "activation_num_first_layer": [None, "tanh"],
    "output_activation": ["sigmoid"],
    "output_dim": [1],
    "batch_size": [16, 32, 64, 128, 256, 1024, 2048, 4096, 8192],
}


class XplaiNetWorker(AbstractWorker):
    def __init__(self, *args, **kwargs):
        self.params = kwargs.pop("params")
        super().__init__(*args, **kwargs)

    #     self.X_train = X_train
    #     self.y_train = y_train
    #     self.X_valid = X_valid
    #     self.y_valid = y_valid
    #     self.X_test = X_test
    #     self.y_test = y_test

    #     self.model_params_keys = self.get_model_params_keys()

    def get_model_params_keys(self):
        return [
            # "params"
            "lconv_dim",
            "lconv_num_dim",
            "emb_size",
            "activation_num_first_layer",
            "output_activation",
            "output_dim",
            # "batch_size"
        ]

    #     (
    # params,
    # lconv_dim=[4],
    # lconv_num_dim=[8],
    # emb_size=16,
    # # For this problem, we need "tanh" as first layer, or else to standard scale the data beforehand
    # activation_num_first_layer=None,  # "tanh",
    # output_activation=None,
    # output_dim=1,  # np.unique(y_train).shape[0],
    # return super().get_model_params_keys()  # + ["tree_method"]

    def get_classifier_class(self):
        def build_model_local(
            lconv_dim=[4],
            lconv_num_dim=[8],
            activation=None,
            optimizer=None,
            emb_size=16,
            activation_num_first_layer=None,
            output_activation="sigmoid",
            output_dim=1,
        ):
            return build_model(
                self.params,
                lconv_dim,
                lconv_num_dim,
                activation,
                optimizer,
                emb_size,
                activation_num_first_layer,
                output_activation,
                output_dim,
            )

        return build_model_local

    def get_budget_desc(self):
        return {"place": "fit", "name": "epochs", "type": "int"}

    def format_train_valid(self):
        return {
            "x": self.X_train,
            "y": self.y_train.reshape(-1, 1),
            "validation_data": (
                self.X_valid,
                self.y_valid.reshape(-1, 1),
            ),
        }
        # return {
        #     "X": self.X_train,
        #     "y": self.y_train,
        #     "eval_set": [(self.X_valid, self.y_valid)],
        # }

    @staticmethod
    def get_grid(custom_config):
        grid = deepcopy(DEFAULT_GRID)
        for key, value in custom_config.items():
            if key in DEFAULT_GRID:
                grid[key] = value

        # if torch.cuda.is_available():
        #     grid["tree_method"] = ["gpu_hist"]
        return grid

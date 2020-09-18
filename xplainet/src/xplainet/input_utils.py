import numpy as np
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor as PoolExecutor

from xplainet.safe_label_encoder import SafeLabelEncoder
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import QuantileTransformer, KBinsDiscretizer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion

def preproc_dataset(train_df, target=None, ids=None, params=None):
    if params is None:
        n_unique = train_df.nunique()

    params = params if params is not None else {}
    to_ignore = []

    if target is not None:
        to_ignore.append(target)

    if ids is not None:
        to_ignore.extend(ids)

    if "constant_cols" not in params:
        constant_cols = train_df.columns[n_unique <= 1]
        constant_cols = list(set(constant_cols.tolist()) - set(to_ignore))
        params["constant_cols"] = constant_cols

    if "bool_cols" not in params:
        bool_cols = train_df.columns[n_unique == 2]
        bool_cols = list(set(bool_cols.tolist()) - set(to_ignore))
        params["bool_cols"] = bool_cols
    if "num_cols" not in params:
        large_int = train_df[train_df.columns[(train_df.dtypes != "object")]].max() > 1e14
        num_cols = list(
            set(
                train_df.columns[
                   (n_unique > 2) & ((n_unique / train_df.shape[0]) >  0.05) & (train_df.dtypes != "object")
                ].tolist()
            )
            - set(large_int.index[large_int.values].tolist())
            - set(to_ignore)
        )
        params["num_cols"] = num_cols

    if "cat_cols" not in params:
        cat_cols = list(
            set(train_df.columns.tolist())
            - set(params["num_cols"])
            - set(params["bool_cols"])
            - set(params["constant_cols"])
            - set(to_ignore)
        )
        params["cat_cols"] = cat_cols

    # Let's handle numeric columns
    input_num_values = train_df[params["num_cols"]].values
    X_num_values = np.zeros(
        (input_num_values.shape[0], input_num_values.shape[1]), dtype="float" #  3 * 
    )

    if "num_encoder" not in params:
        #  Let's calculate fillna for num columns
        fillna_values = (
            train_df[params["num_cols"]].min() - train_df[params["num_cols"]].std() / 10
        )
        params["num_encoder"] = []
        for i in range(len(params["num_cols"])):
            enc = FeatureUnion(
                [
                    (
                        # "fillna",
                        # SimpleImputer(strategy="constant", fill_value=fillna_values[i]),
                        "fillna",
                        Pipeline(
                            [
                                (
                                    "fillna",
                                    SimpleImputer(
                                        strategy="constant", fill_value=fillna_values[i]
                                    ),
                                ),
                                # ("scaler", StandardScaler()),
                            ]
                        ),
                    ),
                    # ("indicator", MissingIndicator(features="all")),
                    # (
                    #     "quantile",
                    #     Pipeline(
                    #         [
                    #             ("fillna", SimpleImputer(strategy="median"),),
                    #             ("quantile", QuantileTransformer()),
                    #         ]
                    #     ),
                    # ),
                    # (
                    #     "bins",
                    #     Pipeline(
                    #         [
                    #             ("fillna", SimpleImputer(strategy="median"),),
                    #             ("bins", KBinsDiscretizer(25, encode="ordinal")),
                    #             ("scaler", StandardScaler()),
                    #         ]
                    #     ),
                    # ),
                ]
            )
            enc.fit(input_num_values[:, i].reshape(-1, 1))
            # enc_list.append(enc)

            params["num_encoder"].append(enc)

    for i, enc in enumerate(params["num_encoder"]):
        # print(enc.transform(input_num_values[:, i].reshape(-1, 1)).reshape(-1, 4).shape)
        X_num_values[:, i :  (i + 1)] = (  #3  *
            enc.transform(input_num_values[:, i].reshape(-1, 1))
            # .reshape(-1, 3)
            .astype("float")
        )

    # Let's handle boolean columns
    X_bool_values = train_df[params["bool_cols"]].values

    if "bool_encoder" not in params:
        params["bool_encoder"] = []
        for i in range(len(params["bool_cols"])):
            enc = SafeLabelEncoder()
            enc.fit(X_bool_values[:, i].reshape(-1))
            params["bool_encoder"].append(enc)

    for i, enc in enumerate(params["bool_encoder"]):
        X_bool_values[:, i] = (
            enc.transform(X_bool_values[:, i].reshape(-1)).reshape(-1).astype("uint")
        )

    # #  For cat cols, let's strip spaces
    # X_cat_values = np.char.strip(train_df[params["cat_cols"]].values.astype("str"))
    # # Now, let's calculate the number of "channels" needed (max string length)
    # if "nb_channels" not in params and len(cat_cols) > 0:
    #     params["nb_channels"] = np.vectorize(len)(X_cat_values).max()
    # elif "nb_channels" not in params:
    #     params["nb_channels"] = 0
    # # Finally, let's transform it into 1d array
    # X_cat_values = do_parallel_numpy(
    #     line_to_2darray, [X_cat_values], [params["nb_channels"]]
    # )
    # X_cat_values = train_df[params["cat_cols"]].values.astype("str")
    input_cat_values = train_df[params["cat_cols"]].values.astype("str")
    X_cat_values = np.zeros(
        (input_cat_values.shape[0], input_cat_values.shape[1]), dtype="uint"
    )

    if "cat_encoder" not in params:
        #  Let's calculate fillna for num columns
        # fillna_values = (
        #     train_df[params["cat_cols"]].min() - train_df[params["num_cols"]].std() / 10
        # )
        params["max_nb"] = -1
        params["cat_encoder"] = []
        for i in range(len(params["cat_cols"])):
            enc = SafeLabelEncoder()
            enc.fit(input_cat_values[:, i].reshape(-1))
            # enc_list.append(enc)
            params["max_nb"] = max(params["max_nb"], len(enc.classes_))
            params["cat_encoder"].append(enc)

    for i, enc in enumerate(params["cat_encoder"]):
        # print(enc.transform(input_num_values[:, i].reshape(-1, 1)).reshape(-1, 4).shape)
        X_cat_values[:, i : i + 1] = (
            enc.transform(input_cat_values[:, i].reshape(-1))
            .reshape(-1, 1)
            .astype("uint")
        )

    X_bool_values = X_bool_values.astype("uint8")

    to_return = []
    if len(params["bool_cols"]) > 0:
        to_return.append(X_bool_values)
    if len(params["num_cols"]) > 0:
        to_return.append(X_num_values)
    if len(params["cat_cols"]) > 0:
        to_return.append(X_cat_values)

    return to_return, params

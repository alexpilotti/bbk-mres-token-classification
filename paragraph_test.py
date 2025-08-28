import concurrent
import gzip
import io
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import Paragraph
from Paragraph import dataset
from Paragraph.model import EGNN_Model
from Paragraph.predict import get_dataloader, evaluate_model
from Paragraph import utils
import requests
from sklearn import metrics
import torch
from torch.utils import data as torch_data

LOG = logging.getLogger(__name__)


A3TO1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


REGIONS = {
    "FR1": [1, 26],
    "CDR1": [27, 38],
    "FR2": [39, 55],
    "CDR2": [56, 65],
    "FR3": [66, 104],
    "CDR3": [105, 117],
    "FR4": [118, 129]
    }


CHAIN_COLUMNS = {"H": "heavy", "L": "light"}


def _filter_list(l, exclude_idxs):
    return [item for i, item in enumerate(l) if i not in exclude_idxs]


class IMGTParagraphDataset(dataset.ParagraphDataset):
    def __init__(self, pdb_H_L_csv, pdb_folder_path, search_area, data, chain):
        super().__init__(pdb_H_L_csv, pdb_folder_path, search_area)
        self._data = data
        self._chain = chain


    def _load_pdb_data(self, index):

        # read in data from csv
        pdb_code = self.df_key.iloc[index]["pdb_code"]

        # read in and process imgt numbered pdb file - keep all atoms
        pdb_path = os.path.join(self.pdb_folder_path, pdb_code + ".pdb")
        df = utils.format_pdb(pdb_path)

        chain_id = self.df_key.iloc[index][f"{self._chain}_id"]

        row = self._data[(self._data.pdb == pdb_code) & (
            self._data[CHAIN_COLUMNS[self._chain]] == chain_id)].iloc[0]
        pdb_numbering = row[f"pdb_numbering_{self._chain}"].split(",")
        IMGT_numbering = row[f"positions_{self._chain}"].split(",")
        pdb_numbering_map = {val: idx for idx, val in enumerate(pdb_numbering)}

        def _get_imgt(pos):
            idx = pdb_numbering_map.get(pos)
            if idx:
                return IMGT_numbering[idx]
            else:
                print(f"Missing PDB residue position in VCAb data: {pos}")

        df.Res_Num = df.Res_Num.apply(lambda x: _get_imgt(x))
        # Remove all rows with no IMGT position
        df = df[~df.Res_Num.isna()]

        return df


def _load_data():
    df = pd.read_parquet('tokens_data.parquet')
    #df = df[df.dataset == "test"]
    df = df[df.dataset == "train"]
    df[["pdb", "heavy", "light"]] = df["iden_code"].str.extract(r"([^_]+)_([0-9a-zA-Z])([0-9a-zA-Z])")

    # Remove entries not compatible with legacy PDB format
    df = df[~df.pdb.isin(["3whe", "5t3x", "7ew5", "7u0p", "7u0q", "7ums",
                          "7ydi", "8dcm", "8dv6", "8dzw", "8gpu", "8hc2",
                          "8iow", "8ivw", "8ivx", "8pn0"])]

    # Entries causing Paragraph to fail
    df = df[~df.pdb.isin(["6axk"])]

    #df = df[df.pdb == "4fqk"]

    return df

def _fetch_pdb(pdb_id):
    base_dir = "pdbs"
    print(f"{pdb_id}\n")

    out_path = os.path.join(base_dir, f"{pdb_id.lower()}.pdb")
    if not os.path.exists(out_path):
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb.gz"

        print(f"Downloading: {pdb_id}")
        r = requests.get(url)
        r.raise_for_status()
        with gzip.open(io.BytesIO(r.content)) as gz:
            with open(out_path, "wb") as f:
                f.write(gz.read())
    return out_path


def _fetch_pdbs(data):
    pdb_ids = sorted(data.pdb.unique().tolist())
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for result in ex.map(_fetch_pdb, pdb_ids):
            print(result)


def _save_csv(data, csv_path, chain):
    if "H" not in chain:
        data.heavy = ""
    if "L" not in chain:
        data.light = ""
    data[["pdb", "heavy", "light"]].to_csv(csv_path, index=False, header=False)

def _run_paragraph(chain, data, pdb_folder_path, pdb_H_L_csv,
                   predictions_output_path):
    src_path = os.path.dirname(Paragraph.__file__)

    trained_model_path = os.path.join(src_path, "trained_model")

    if chain == "H":
        model_file = "pretrained_weights_heavy.pt"
    elif chain == "L":
        model_file = "pretrained_weights_light.pt"
    else:
        model_file = "pretrained_weights.pt"

    saved_model_path = os.path.join(trained_model_path, model_file)

    # fix seed
    seed = 42
    torch.manual_seed(seed)

    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device being used: {device}\n")

    # examine dataset object contents

    search_area = "IMGT_CDRplus2"

    ds = IMGTParagraphDataset(pdb_H_L_csv, pdb_folder_path, search_area, data,
                              chain)

    (feats, coors, edges), (pdb_code, AAs, AtomNum, chains, chain_types, IMGT, x, y, z) = ds.__getitem__(0)
    print(pdb_code)
    print("Co-ordinate dimensions: \t{}".format(coors.shape))
    print("Node feature dimensions: \t{}".format(feats.shape))
    print("Edge feature dimensions: \t{}\n".format(edges.shape))

    df = pd.DataFrame(zip(AAs, AtomNum, chain, IMGT, x, y, z),
                    columns=["AAs", "AtomNum", "chain", "IMGT", "x", "y", "z"])
    print(df.head())

    # test model architecture

    # use small number of features and samples so results can be easily read manually
    num_samples = 3
    num_feats = 6

    # one graph layer and 1 hidden linear layer
    graph_hidden_layer_output_dims = [num_feats]
    linear_hidden_layer_output_dims = [int(0.5*num_feats)]

    # create dummy data and examine this
    feats = torch.rand((num_samples, num_feats), device=device).unsqueeze_(0)
    coors = torch.rand((num_samples, 3), device=device).unsqueeze_(0)
    edges = torch.rand((num_samples, num_samples, 1), device=device).unsqueeze_(0)
    print("Input features:\n", feats)

    # create our model
    #dummy_net = EGNN_Model(num_feats = num_feats,
    #                    graph_hidden_layer_output_dims = graph_hidden_layer_output_dims,
    #                    linear_hidden_layer_output_dims = linear_hidden_layer_output_dims)

    # add model to gpu if possible
    #dummy_net = dummy_net.to(device)

    # pass our data through our model and examine the new embeddings
    #feats = dummy_net.forward(feats, coors, edges)
    #print("\nOutput features (i.e. node predictions):\n", feats)

    # network dims used in pre-trained model

    num_feats = 22  # 20D one-hot encoding of AA type and 2D one-hot encoding of chain ID
    graph_hidden_layer_output_dims = [num_feats]*6
    linear_hidden_layer_output_dims = [10]*2
    # weights and predictions paths


    #dl = get_dataloader(pdb_H_L_csv, pdb_folder_path)

    batch_size = 1
    ds = IMGTParagraphDataset(pdb_H_L_csv, pdb_folder_path, search_area, data,
                              chain)
    dl = torch_data.DataLoader(dataset=ds, batch_size=batch_size)

    saved_net = EGNN_Model(num_feats = num_feats,
                        graph_hidden_layer_output_dims = graph_hidden_layer_output_dims,
                        linear_hidden_layer_output_dims = linear_hidden_layer_output_dims)

    try:
        saved_net.load_state_dict(torch.load(saved_model_path))
    except RuntimeError:
        saved_net.load_state_dict(torch.load(saved_model_path, map_location=torch.device('cpu')))
    saved_net = saved_net.to(device)
    print("Evaluating using weight file:\n{}\n".format(saved_model_path.split("Paragraph")[-1]))
    start_time = time.time()

    detailed_record_df = evaluate_model(model = saved_net,
                                        dataloader = dl,
                                        device = device)
    detailed_record_df.to_csv(predictions_output_path, index=False)

    print("Results saved to:\n{}\n".format(predictions_output_path.split("Paragraph")[-1]))
    print("Total time to evaluate against test-set {:.3f}s".format(time.time()-start_time))


def _get_num_pos(pos):
    """Needed for positions with insertion codes, e.g. 110A."""
    return int("".join(c for c in pos if c.isdigit()))


def _get_all_fr_positions():
    fr_positions = []
    for i in range(1, 5):
        fr_min, fr_max = REGIONS[f"FR{i}"]
        fr_positions += list(range(fr_min, fr_max + 1))
    return fr_positions


def _process_paragraph_output(data, chain, zero_fr_positions,
                              predictions_output_path):
    df_pred = pd.read_csv(predictions_output_path)
    #df_pred["A"] = df_pred.AA.apply(lambda x: A3TO1[x])
    df_pred["label"] = df_pred.pred.apply(lambda x: 1 if x >= 0.734 else 0)

    results_df = data[["iden_code", f"labels_{chain}", f"positions_{chain}"]]
    results_df = results_df.rename(
        columns={f"labels_{chain}": "labels",
                 f"positions_{chain}":  "positions"})
    results_df["predicted_labels"] = ""
    results_df["probs"] = ""

    fr_positions = _get_all_fr_positions()

    for idx in data.index:
        row = data.loc[idx]
        print(f"Processing PDB: {row.pdb}")

        pred_labels = []
        probs = []


        for c in CHAIN_COLUMNS.keys():
            if c in chain :
                chain_id = row[CHAIN_COLUMNS[c]]
            else:
                continue

            df = df_pred[(df_pred.pdb == row.pdb) &
                        (df_pred.chain_type == c) &
                        (df_pred.chain_id == chain_id)]

            imgt_numbering = row[f"positions_{c}"].split(",")
            labels = list(map(int, row[f"labels_{c}"].split(",")))
            assert len(imgt_numbering) == len(labels)

            for idx2, pos in enumerate(imgt_numbering):
                pred_label = -100
                prob = 0
                if pos != "":
                    df_pos = df[df.IMGT == pos]
                    if len(df_pos):
                        row_pred = df_pos.iloc[0]
                        vcab_res = row[f"sequence_{c}"][idx2]
                        paragraph_res = A3TO1[row_pred.AA]
                        if paragraph_res == vcab_res:
                            pred_label = row_pred.label
                            prob = row_pred.pred
                        else:
                            print(f"WARNING: {row.pdb} {pos} mismatch, VCAb "
                                  f"{vcab_res}, Paragraph: {paragraph_res}")
                            pred_label = -100
                    elif zero_fr_positions:
                        # Set 0 for all missing FR positions
                        if _get_num_pos(pos) in fr_positions:
                            pred_label = 0

                probs.append(prob)
                pred_labels.append(pred_label)

            df_extra_pos = df[~df.IMGT.isin(imgt_numbering)]
            if len(df_extra_pos):
                print(f"Extra predicted positions not in data: {df_extra_pos}")

        results_df.loc[idx, "predicted_labels"] = ",".join(
            str(l) for l in pred_labels)
        results_df.loc[idx, "probs"] = ",".join(
            str(l) for l in probs)

    return results_df


def _false_positive_rate(y_true, y_pred):
    tn, fp, _, _ = metrics.confusion_matrix(
        y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn)


def _compute_metrics(df):
    labels = []
    predictions = []

    labs = df.labels.str.split(",").apply(
        lambda l: list(map(int, l)))
    preds = df.predicted_labels.str.split(",").apply(
        lambda l: list(map(int, l)))
    probs = df.probs.str.split(",").apply(
        lambda l: list(map(float, l)))

    df1 = pd.concat(
        {"label": labs, "pred": preds, "prob": probs}, axis=1).apply(
            pd.Series.explode).reset_index(drop=True)

    ## Skip all cases where either the labels or the predictions are missing
    #df1 = df1[(df1.label != - 100) & (df1.pred != - 100)]
    # Skip all cases where the labels are missing
    df1 = df1[(df1.label != - 100)]
    labels = df1.label.tolist()
    predictions = df1.pred.tolist()
    probs = df1.prob.tolist()

    report = metrics.classification_report(
        y_true=labels,
        y_pred=predictions,
        zero_division=0,
        digits=4,
        output_dict=True
    )

    if len(set(labels)) > 1:
        # AUC is undefined if there's a single class
        auc = metrics.roc_auc_score(labels, probs)
    else:
        auc = 0

    report["apr"] = metrics.average_precision_score(
        labels, probs, pos_label=1)
    report["balanced_accuracy"] = metrics.balanced_accuracy_score(
        labels, predictions)
    report["auc"] = auc
    report["mcc"] = metrics.matthews_corrcoef(labels, predictions)
    report["fpr"] = _false_positive_rate(labels, predictions)
    report["f1_weighted"] = metrics.f1_score(
        labels, predictions, average="weighted")

    return report


def _filter_region_data(data, region):
    if region not in REGIONS:
        raise Exception(f'Invalid region name: "{region}". '
                        f'Valid options are: {", ".join(REGIONS.keys())}')

    region_range = range(*REGIONS[region])

    LOG.info(f"Retrieving residues in region {region}, "
             f"between positions: {region_range[0]}-{region_range[-1]}")

    data["positions_num"] = data["positions"].apply(
        lambda x: list(map(_get_num_pos, x.split(","))))

    data.labels = df.labels.str.split(",").apply(
        lambda l: list(map(int, l)))
    data.predicted_labels = df.predicted_labels.str.split(",").apply(
        lambda l: list(map(int, l)))
    data.probs = df.probs.str.split(",").apply(
        lambda l: list(map(float, l)))

    #data["sequence"] = data.apply(
    #    lambda row: "".join(
    #        row["sequence"][i] for i, pos in enumerate(row["positions_num"])
    #        if pos in region_range),
    #    axis=1)

    data["labels"] = data.apply(
        lambda row:
            ",".join([str(row["labels"][i]) for i, pos in enumerate(row["positions_num"])
             if pos in region_range]),
        axis=1)

    data["predicted_labels"] = data.apply(
        lambda row:
            ",".join([str(row["predicted_labels"][i]) for i, pos in enumerate(row["positions_num"])
             if pos in region_range]),
        axis=1)

    data["probs"] = data.apply(
        lambda row:
            ",".join([str(row["probs"][i]) for i, pos in enumerate(row["positions_num"])
             if pos in region_range]),
        axis=1)

    data["positions"] = data["positions"].apply(
        lambda x: ",".join(
            [pos for pos in x.split(",")
             if _get_num_pos(pos) in region_range]))

    data = data.drop('positions_num', axis=1)

    return data


def _save_metrics(df, region, report_path):

    if region != "FULL":
        df = _filter_region_data(df.copy(), region)

    report = _compute_metrics(df)

    report['model_name'] = "Paragraph"
    report['model_path'] = ""
    report['num_parameters'] = 0
    report['test_loss'] = 0

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)


if __name__ == "__main__":
    chain = sys.argv[1] if len(sys.argv) > 1 else "H"

    zero_fr_positions = True

    base_dir = os.getcwd()
    pdb_folder_path = os.path.join(base_dir, "pdbs")
    pdb_H_L_csv = os.path.join(base_dir, f"data_{chain}.csv")
    predictions_output_path = os.path.join(base_dir, f"predictions_{chain}.csv")

    data = _load_data()
    _fetch_pdbs(data)
    _save_csv(data, pdb_H_L_csv, chain)
    _run_paragraph(chain, data, pdb_folder_path, pdb_H_L_csv,
                   predictions_output_path)
    out_data_path = f"token_prediction_Paragraph_{chain}.parquet"
    df = _process_paragraph_output(data, chain, zero_fr_positions,
                                   predictions_output_path)
    df.to_parquet(out_data_path, index=False)

    df = pd.read_parquet(out_data_path)
    for region in list(REGIONS.keys()) + ["FULL"]:
        report_path = f"token_predict_metrics_Paragraph_{chain}_FULL_{region}_PT.json"
        _save_metrics(df, region, report_path)

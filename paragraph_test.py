import concurrent
import gzip
import io
import json
import os
import time

import pandas as pd
import Paragraph
from Paragraph.dataset import ParagraphDataset
from Paragraph.model import EGNN_Model
from Paragraph.predict import get_dataloader, evaluate_model
import requests
from sklearn import metrics
import torch

A3TO1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


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

def _run_paragraph(chain):
    src_path = os.path.dirname(Paragraph.__file__)
    base_dir = os.getcwd()

    trained_model_path = os.path.join(src_path, "trained_model")

    if chain == "H":
        model_file = "pretrained_weights_heavy.pt"
    elif chain == "L":
        model_file = "pretrained_weights_light.pt"
    else:
        model_file = "pretrained_weights.pt"

    saved_model_path = os.path.join(trained_model_path, model_file)

    pdb_folder_path = os.path.join(base_dir, "pdbs")
    pdb_H_L_csv = os.path.join(base_dir, "data.csv")
    predictions_output_path = os.path.join(base_dir, "predictions.csv")

    # fix seed
    seed = 42
    torch.manual_seed(seed)

    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device being used: {device}\n")

    # examine dataset object contents

    ds = ParagraphDataset(pdb_H_L_csv=pdb_H_L_csv, pdb_folder_path=pdb_folder_path)

    (feats, coors, edges), (pdb_code, AAs, AtomNum, chain, chain_type, IMGT, x, y, z) = ds.__getitem__(0)
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


    dl = get_dataloader(pdb_H_L_csv, pdb_folder_path)

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


def _process_paragraph_output(data, chain):
    df_pred = pd.read_csv("predictions.csv")
    #df_pred["A"] = df_pred.AA.apply(lambda x: A3TO1[x])
    df_pred["label"] = df_pred.pred.apply(lambda x: 1 if x >= 0.734 else 0)

    results_df = data[["iden_code", f"labels_{chain}", f"positions_{chain}"]]
    results_df = results_df.rename(
        columns={f"labels_{chain}": "labels",
                 f"positions_{chain}":  "positions"})
    results_df["predicted_labels"] = ""

    for idx in data.index:
        row = data.loc[idx]
        pred_labels = []

        chain_columns = {"H": "heavy", "L": "light"}

        for c in chain_columns.keys():
            if c in chain :
                chain_id = row[chain_columns[c]]
            else:
                continue

            df = df_pred[(df_pred.pdb == row.pdb) &
                        (df_pred.chain_type == c) &
                        (df_pred.chain_id == chain_id)]

            pdb_numbering = row[f"pdb_numbering_{c}"].split(",")
            labels = list(map(int, row[f"labels_{c}"].split(",")))
            assert len(pdb_numbering) == len(labels)

            for idx2, pos in enumerate(pdb_numbering):
                pred_label = -100
                if pos != "":
                    df_pos = df[df.IMGT == pos]
                    if len(df_pos):
                        row_pred = df_pos.iloc[0]
                        vcab_res = row[f"sequence_{c}"][idx2]
                        paragraph_res = A3TO1[row_pred.AA]
                        if paragraph_res == vcab_res:
                            pred_label = row_pred.label
                        else:
                            print(f"WARNING: {row.pdb} {pos} mismatch, VCAb "
                                  f"{vcab_res}, Paragraph: {paragraph_res}")
                            pred_label = -100
                pred_labels.append(pred_label)

            df_extra_pos = df[~df.IMGT.isin(pdb_numbering)]
            if len(df_extra_pos):
                print(f"Extra predicted positions not in data: {df_extra_pos}")

        results_df.loc[idx, "predicted_labels"] = ",".join(
            str(l) for l in pred_labels)

    return results_df


def _compute_metrics(df):
    labels = []
    predictions = []

    l_list = df.labels.str.split(",").apply(
        lambda l: list(map(int, l)))
    p_list = df.predicted_labels.str.split(",").apply(
        lambda l: list(map(int, l)))

    df1 = pd.concat(
        {"label": l_list, "pred": p_list}, axis=1).apply(
            pd.Series.explode).reset_index(drop=True)

    # Skip all cases where either the labels or the predictions are missing
    df1 = df1[(df1.label != - 100) & (df1.pred != - 100)]
    labels = df1.label.tolist()
    predictions = df1.pred.tolist()

    report = metrics.classification_report(
        y_true=labels,
        y_pred=predictions,
        zero_division=0,
        digits=4,
        output_dict=True
    )

    return report


if __name__ == "__main__":
    chain = "H"

    data = _load_data()
    _fetch_pdbs(data)
    _save_csv(data, "data.csv", chain)
    _run_paragraph(chain)
    df = _process_paragraph_output(data, chain)
    report = _compute_metrics(df)

    report['model_name'] = "Paragraph"
    report['model_path'] = ""
    report['num_parameters'] = 0
    report['test_loss'] = 0

    report_path = f"prediction_metrics_Paragraph_{chain}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    df.to_parquet(
        f"token_prediction_Paragraph_{chain}.parquet", index=False)

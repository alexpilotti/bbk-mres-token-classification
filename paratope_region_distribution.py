import json
import logging

import pandas as pd

LOG = logging.getLogger(__name__)


REGIONS = {
    "FR1": [1, 26],
    "CDR1": [27, 38],
    "FR2": [39, 55],
    "CDR2": [56, 65],
    "FR3": [66, 104],
    "CDR3": [105, 117],
    "FR4": [118, 129]
    }


def _get_num_pos(pos):
    """Needed for positions with insertion codes, e.g. 110A."""
    return int("".join(c for c in pos if c.isdigit()))


def _filter_region_data(data, region):
    if region not in REGIONS:
        raise Exception(f'Invalid region name: "{region}". '
                        f'Valid options are: {", ".join(REGIONS.keys())}')

    region_range = range(*REGIONS[region])

    LOG.info(f"Retrieving residues in region {region}, "
             f"between positions: {region_range[0]}-{region_range[-1]}")

    data["positions_num"] = data["positions"].apply(
        lambda x: list(map(_get_num_pos, x.split(","))))

    data["sequence"] = data.apply(
        lambda row: "".join(
            row["sequence"][i] for i, pos in enumerate(row["positions_num"])
            if pos in region_range),
        axis=1)

    data["labels"] = data.apply(
        lambda row:
            [row["labels"][i] for i, pos in enumerate(row["positions_num"])
             if pos in region_range],
        axis=1)

    data["positions"] = data["positions"].apply(
        lambda x: ",".join(
            [pos for pos in x.split(",")
             if _get_num_pos(pos) in region_range]))

    return data


def _get_paratope_residues_per_region(data, dataset, chain):
    data = data[data.dataset == dataset]

    data = data.rename(columns={
        f"positions_{chain}": "positions",
        f"labels_{chain}": "labels",
        f"sequence_{chain}": "sequence"})

    data["labels"] = data["labels"].apply(
        lambda x: list(map(int, x.split(","))))

    paratope_totals = {}
    paratope_perc = {}

    for region in REGIONS.keys():
        region_data = _filter_region_data(data.copy(), region)
        paratope_totals[region] = int(region_data.labels.apply(
            lambda x: x.count(1)).sum())

        region_length = REGIONS[region][1] - REGIONS[region][0] + 1
        paratope_perc[region] = paratope_totals[region] / (
            region_length * len(region_data))

    return {"totals": paratope_totals, "perc_in_region": paratope_perc}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data = pd.read_parquet('tokens_data.parquet')

    paratope_totals = {}

    for dataset in ["train", "test"]:
        paratope_totals[dataset] = {}
        for chain in ["H", "L"]:
            LOG.info(f"dataset: {dataset}, chain: {chain}")
            paratope_totals[
                dataset][chain] = _get_paratope_residues_per_region(
                    data, dataset, chain)

    LOG.info(paratope_totals)

    with open("paratope_residues_per_region.json", "w") as f:
        json.dump(paratope_totals, f)

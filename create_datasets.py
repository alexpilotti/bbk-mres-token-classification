import itertools
import logging

import numpy as np
import pandas as pd
import sqlalchemy


logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

# https://www.imgt.org/IMGTScientificChart/Nomenclature/IMGT-FRCDRdefinition.html
#CDR3_range = [105, 117]

insert_missing_residues = True

engine = sqlalchemy.create_engine('sqlite:///vcab.db', echo=False)
with engine.connect() as conn:
    df = pd.read_sql(
        "select m.iden_code, v.species, m.chain, m.anarci_pos, m.anarci_ins, "
        "m.residue, m.pdb_numbering, cast(m.if_interface_res as integer) as "
        "if_interface_res from mapping m join vcab v on m.iden_code = "
        "v.iden_code where m.anarci_pos is not null "
        #f"and m.anarci_pos >= {CDR3_range[0] - 1} "
        #f"and m.anarci_pos <= {CDR3_range[1] - 1} "
        "order by m.iden_code, m.chain, m.anarci_pos, m.anarci_ins", conn)

# Used for test
#df = df[df.iden_code == "1a3r_HL"]

df["dataset"] = df["Species"].apply(
    lambda x: "train" if x == "Homo_Sapiens" else "test")

df['if_interface_res'] = df['if_interface_res'].astype('Int64')

df.loc[df['if_interface_res'].isna(), 'if_interface_res'] = -100

max_pos = df["anarci_pos"].max()
#max_pos = df["anarci_pos"].apply(
#    lambda x: x if isinstance(x, int) else int(x[:-1])).max()

full_range = pd.Series(range(1, max_pos + 1), name='anarci_pos')

if insert_missing_residues:
    df3 = pd.DataFrame()

    unique_combinations = df[['iden_code', 'chain']].drop_duplicates()
    #all_combinations = pd.DataFrame(
    #    list(itertools.product(unique_combinations['iden_code'], unique_combinations['chain'], full_range)),
    #    columns=['iden_code', 'chain', 'anarci_pos']
    #)

    for row in unique_combinations.itertuples():
        print(f"{row.iden_code} {row.chain}")
        df1 = df[(df["iden_code"] == row.iden_code) &
                 (df["chain"] == row.chain)]
        missing_pos = full_range[~full_range.isin(df1["anarci_pos"])]
        if len(missing_pos) == 0:
            continue
        first_row = df1.iloc[0]
        df2 = pd.DataFrame(missing_pos)
        df2["residue"] = "X"
        df2["if_interface_res"] = -100
        for c in [c for c in df1.columns if c not in df2.columns]:
            df2[c] = first_row[c]
        logger.info(f"{row.iden_code} {row.chain} inserting "
                    f"missing positions: {len(df2)}")
        df3 = pd.concat([df3, df2])

    df = pd.concat([df, df3])
    df = df.sort_values(by=['iden_code', 'chain', 'anarci_pos'],
                        ascending=[True, True, True])

#df_complete = pd.merge(all_combinations, df, on=['iden_code', 'chain', 'anarci_pos'], how='left')

df["anarci_pos_ins"] = (df["anarci_pos"].astype(str) +
                        df["anarci_ins"].fillna(""))

df["pdb_numbering"] = df["pdb_numbering"].fillna("")

# group_keys to retain sorting
df = df.groupby(
        ["iden_code", "Species", "dataset", "chain"], group_keys=False).agg({
    'anarci_pos_ins': lambda x: ','.join(map(str, x)),
    'residue': lambda x: ''.join(x),
    'if_interface_res': lambda x: ','.join(map(str, x)),
    'pdb_numbering': lambda x: ','.join(map(str, x))
}).reset_index()

df = df.rename(columns={
    'anarci_pos_ins': 'positions',
    'if_interface_res': 'labels',
    'residue': 'sequence'})

pivoted_df = df.pivot(index=['iden_code', 'Species', 'dataset'],
                      columns='chain',
                      values=['positions', 'pdb_numbering', 'sequence', 'labels'])
pivoted_df.columns = [f"{col[0]}_{col[1]}" for col in pivoted_df.columns]
pivoted_df = pivoted_df.reset_index()
df = pivoted_df

logger.info(f"Initial sequences: {len(df)}")

# TODO: find a clean way to remove improbable sequences
df = df[~df["positions_H"].str.contains("O", na=False)]

# Remove cases where either chain H or L is missing
missing_a_chain = df["sequence_H"].isna() | df["sequence_L"].isna()
logger.info("Removing sequences with missing H or L chains: "
            f"{len(df[missing_a_chain])}")
df = df[~missing_a_chain]

df["sequence_HL"] = df["sequence_H"] + df["sequence_L"]

df["labels_HL"] = df["labels_H"] + "," + df["labels_L"]

positions_H = df["positions_H"].apply(
    lambda x: ",".join(f"H{x}" for x in x.split(",")))
positions_L = df["positions_L"].apply(
    lambda x: ",".join(f"L{x}" for x in x.split(",")))

df["positions_HL"] = positions_H + "," + positions_L

duplicates = df['sequence_HL'].duplicated()
logger.info(f"Removing duplicated sequences: {len(df[duplicates])}")
df = df[~duplicates]

#df = df.reset_index()

# Ignore rows having one or more null labels
#df_filtered = df[~df['labels'].str.contains('NULL', case=False, na=False)]
#logger.info("Rows removed due to one or more NULL label: "
#            f"{len(df) - len(df_filtered)}")

print(df.groupby(['dataset']).size().reset_index(name='Count'))

logger.info(f"Final number of sequences: {len(df)}")
df.to_parquet("tokens_data.parquet")

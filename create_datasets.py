import logging

import numpy as np
import pandas as pd
import sqlalchemy


logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

# https://www.imgt.org/IMGTScientificChart/Nomenclature/IMGT-FRCDRdefinition.html
CDR3_range = [105, 117]

engine = sqlalchemy.create_engine('sqlite:///vcab.db', echo=False)
with engine.connect() as conn:
    df = pd.read_sql(
        "select m.iden_code, v.species, m.chain, m.anarci_pos, residue, "
        "cast(m.if_interface_res as integer) as if_interface_res "
        "from mapping m join vcab v on m.iden_code = v.iden_code "
        "where m.anarci_pos is not null and "
        f"m.anarci_pos >= {CDR3_range[0] - 1} and "
        f"m.anarci_pos <= {CDR3_range[1] - 1} "
        "order by m.iden_code, m.chain, m.anarci_pos", conn)

df["dataset"] = df["Species"].apply(
    lambda x: "train" if x == "Homo_Sapiens" else "test")

df['if_interface_res'] = df['if_interface_res'].astype('Int64')


# TODO: Data comes already sorted from the query, check if this can be removed
df.sort_values(by=['iden_code', 'chain', 'anarci_pos'],
               ascending=[True, True, True])

# group_keys to retain sorting
df = df.groupby(
        ["iden_code", "Species", "dataset", "chain"], group_keys=False).agg({
    'anarci_pos': lambda x: ','.join(map(str, x)),
    'residue': lambda x: ''.join(x),
    'if_interface_res': lambda x: ','.join(
        "NULL" if pd.isna(v) else str(v) for v in x)
}).reset_index()

df = df.rename(columns={
    'anarci_pos': 'positions',
    'if_interface_res': 'labels',
    'residue': 'sequence'})

duplicates = df['sequence'].duplicated()
logger.info(f"Removing duplicated sequences: {len(duplicates)}")
df = df[~duplicates]

# Ignore rows having one or more null labels
df_filtered = df[~df['labels'].str.contains('NULL', case=False, na=False)]
logger.info("Rows removed due to one or more NULL label: "
            f"{len(df) - len(df_filtered)}")

print(df.groupby(['dataset']).size().reset_index(name='Count'))

logger.info(f"Final number of sequences: {len(df_filtered)}")
df_filtered.to_parquet("tokens_data.parquet")

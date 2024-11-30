import pandas as pd
import sqlite3


def get_tokens(species):
    df_out = pd.DataFrame(columns=["pdbid", "iden_code", "species", "H", "L", "H_labels", "L_labels"])

    with sqlite3.connect("vcab.db") as con:
        df_vcab = pd.read_sql_query(
            f"select pdb, HV_seq, LV_Seq, Species from vcab where "
            f"species = '{species}' order by pdb", con)

        for _, row in df_vcab.iterrows():
            pdbid = row["pdb"]
            hv_seq = row["HV_seq"]
            lv_seq = row["LV_seq"]

            df_mapping = pd.read_sql_query(
                f"select * from mapping where iden_code like '{pdbid}_%' order by pdb_chain, chain, text_numbering", con)

            for iden_code in df_mapping.iden_code.unique():
                df = df_mapping[df_mapping["iden_code"] == iden_code]
                chains = ['H', 'L']

                labels = {}
                for chain in chains:
                    seq = hv_seq if chain == 'H' else lv_seq
                    df_chain = df[df["chain"] == chain].sort_values(by="text_numbering", ascending=True)
                    if not len(df_chain):
                        print(f"Chain {chain} missing for {iden_code}, skipping")
                        continue

                    if len(df_chain) != len(seq):
                        print(f"Seq length mismatch. iden_token: {iden_code}, chain: {chain}, Seq: {len(seq)}, mapping rows: {len(df_chain)}, skipping")
                        continue

                    for i in range(0, len(df_chain)):
                        if df_chain.iloc[i]["text_numbering"] != i:
                            raise Exception(f"Missing index {i} for iden_token: {iden_code}, chain: {chain}")

                    chain_labels = ','.join(df_chain["if_interface_res"].fillna('').apply(lambda x: '{:.0f}'.format(x) if isinstance(x, (int, float)) else str(x)))
                    labels[chain] = chain_labels

                if len(labels) != len(chains):
                    continue

                out_row = {"pdbid": pdbid, "iden_code": iden_code, "species": species, "H": hv_seq, "L": lv_seq, "H_labels": labels['H'], "L_labels": labels['L']}
                df_out.loc[-1] = out_row

    return df_out


if __name__ == "__main__":
    df_out = get_tokens("Homo_Sapiens")
    df_out.to_parquet("toekn_classification_data.parquet")
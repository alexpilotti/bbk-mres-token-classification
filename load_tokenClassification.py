import os
import re

import csv
import pandas as pd
import sqlite3

import Bio
from Bio import SeqUtils

from sqlalchemy import create_engine

import mapping

pattern = re.compile(r"([a-zA-Z0-9]+)\_([a-zA-Z0-9])([a-zA-Z0-9])-([a-zA-Z0-9])([a-zA-Z0-9])\_deltaSASA\_rpopsResidue.txt")

engine = create_engine('sqlite:///vcab.db', echo=False)
with engine.connect() as conn:
    vcab = pd.read_csv('final_vcab_with_V_coor.csv')
    vcab.to_sql("vcab", conn, if_exists='append', index=False)

    pops_dir = "pops_result"
    pops_results = os.listdir(pops_dir)
    for pr in pops_results:
        match = pattern.match(pr)
        if match:
            result = match.groups()
            print(f"{pr} - {result}")
            pdbid = result[0]
            iden_chain_h = result[1]
            iden_chain_l = result[2]
            ab_chain = result[3]
            antigen_chain = result[4]
            iden_code = f"{pdbid}_{iden_chain_h}{iden_chain_l}"

            pops_data = pd.read_csv(os.path.join(pops_dir, pr), sep=' ', index_col=0)
            if len(pops_data) == 0:
                print("pops_data is empty")
            else:
                pops_data = mapping.replace_back_T_F_chain (pops_data)
                pops_data = pops_data[pops_data["Chain"] == ab_chain]

                if len(pops_data) == 0:
                    print("Filtered data by Chain is empty")

                pops_data.insert(0, "PDBID", pdbid)
                pops_data.insert(1, "iden_code", iden_code)
                pops_data.insert(2, "iden_chain_H", iden_chain_h)
                pops_data.insert(3, "iden_chain_L", iden_chain_l)
                pops_data.insert(4, "ab_chain", ab_chain)
                pops_data.insert(5, "antigen_chain", antigen_chain)
                pops_data.insert(6, "ResidNe_1", pops_data['ResidNe'].apply(
                    lambda x: Bio.SeqUtils.IUPACData.protein_letters_3to1[x.capitalize()]))

                pops_data.reset_index()
                pops_data.to_sql("pops_results", conn, if_exists='append', index=False)

            chain = "H" if iden_chain_h == ab_chain else "L"
            d_sasa_th = 15

            mapping_results = mapping.mapping_positions(
                iden_code,vcab, f"{chain}V_seq", f"{chain}V_coordinate_seq",
                f"{chain}V_PDB_numbering", "pops_result", "antigen_chain",
                chain, d_sasa_th)

            mapping_results.insert(0, "iden_code", iden_code)
            mapping_results.insert(1, "chain", chain)
            mapping_results.insert(2, "pdb_chain", ab_chain)

            mapping_results.to_sql("mapping", conn, if_exists='append', index=False)
        else:
            raise Exception(f"Can't match: {pr}")

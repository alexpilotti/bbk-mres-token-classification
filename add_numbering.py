import logging

import anarci
import sqlalchemy

logger = logging.getLogger(__name__)


def _get_adj_sequence_numbering(sequence, scheme):
    numbering, chain_type = anarci.number(sequence, scheme=scheme)

    logger.info(f"Sequence: {sequence}")
    logger.info(f"ANARCI sequence: {''.join([r[1] for r in numbering])}")

    adj_seq_numbering = []
    j = 0
    l = [(a, b, c) for  ((a,b), c) in numbering if b != ' ']
    if len(l):
        pass

    for i, r1 in enumerate(sequence):
        while True:
            err_msg = ()
            if j >= len(numbering):
                    logger.warning(
                        f"Residue {r1} at position {i} is beyond the sequence "
                       "returned from ANARCI")
                    adj_seq_numbering.append((None, None))
                    break
            (pos, ins), r2 = numbering[j]
            if ins != ' ' or r1 == r2:
                ins = ins.strip()
                adj_seq_numbering.append((pos, ins if len(ins) else None))
                j += 1
                break
            elif r2 != "-":
                logger.warning(f"Unrecognized residue {r1} at position {i}")
                adj_seq_numbering.append((None, None))
                break
            else:
                j += 1



    assert len(adj_seq_numbering) == len(sequence)
    return adj_seq_numbering, chain_type


logging.basicConfig(level=logging.INFO)

engine = sqlalchemy.create_engine('sqlite:///vcab.db', echo=False)
with engine.connect() as conn:
    conn.execute(sqlalchemy.text("alter table mapping add column anarci_pos int"))
    conn.execute(sqlalchemy.text("alter table mapping add anarci_ins char(1)"))

    #conn.execute(sqlalchemy.text("update mapping set anarci_pos = NULL"))
    conn.commit()

    update_cmd = sqlalchemy.text("update mapping set anarci_pos = :anarci_pos, anarci_ins = :anarci_ins where iden_code = :iden_code and chain = :chain and text_numbering = :seq_pos")
    result = conn.execute(sqlalchemy.text("select iden_code, hv_seq, lv_seq, h_seq, l_seq from vcab order by iden_code"))
    for row in result:
        for chain in ["H", "L"]:
            logger.info(f"iden_code: {row.iden_code}, chain: {chain}")
            adj_seq_numbering, chain_type = _get_adj_sequence_numbering(
                row.HV_seq if chain == "H" else row.LV_seq, "imgt")

            if chain_type != chain:
                raise Exception("ANARCI chain mismatch: {chain_type}  {row.chain}")

            for seq_pos, (anarci_pos, anarci_ins) in enumerate(adj_seq_numbering):
                logger.debug(f"{row.iden_code}, {chain}, {seq_pos}, {anarci_pos} {anarci_ins}")
                conn.execute(update_cmd, {"iden_code": row.iden_code, "chain": chain, "anarci_pos": anarci_pos, "anarci_ins": anarci_ins, "seq_pos": seq_pos})
            conn.commit()



        #n, chain_type = anarci.number(row.HV_seq, scheme="imgt")
        #print(len(n))

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from Bio.Align import substitution_matrices

from Bio import pairwise2
#from Bio.SubsMat import MatrixInfo as matlist
from Bio.SeqUtils import seq1

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

def map_pure_seq_pos_to_aln_pos (pure_pos,aln_seq):
    # return the aln_pos of the corresponding pure_seq
    # pure_pos: the single position of the pure seq, which does not contain the gaps
    # aln_seq: the seq from the aln object, which containing the gaps
    # both the numbering of pure_pos and aln_seq starts from 0
    pure_seq_counter=0
    for aln_pos,r in enumerate(aln_seq):
        if r!="-":
            if pure_seq_counter==pure_pos:
                return aln_pos
            pure_seq_counter+=1

def map_aln_pos_to_pure_seq_pos (aln_pos,aln_seq):
    # return the pure_seq_pos, given the aln_pos and aln_seq
    # the numbering of both aln_pos & pure_seq_pos starts from 0
    str_before_aln_pos=aln_seq[0:aln_pos+1]
    pure_str=str_before_aln_pos.replace("-","")
    ## get the pure seq without gap "-" before aln_pos
    ## thus, the residues at the pure_pos given by this function would either be
    ## the residue at aln_pos or be the residue at the the left side of aln_pos
    ## (in case where character at aln_pos is a gap)

    pure_seq_pos=len(pure_str)-1
    return pure_seq_pos

def replace_back_T_F_chain (o_df):
    df=o_df.copy()
    for c,i in enumerate(df['Chain']):
        if i == 'TRUE':
            #print (c,i)
            df['Chain'][c]='T'
        if i == 'FALSE':
            #print (df['Chain'][c])
            df['Chain'][c]='F'

    return df

def read_pops_file(iden_code,pops_dir,antigen,chainType, d_sasa_th):
    """
    Return popsFile: h_df, l_df, which contains residues with d_SASA >15
    :args antigen: antigen chain ID
    chainType: "H" or "L"

    """
    h,l=iden_code.split('_')[1]
    ab_chain=h
    if chainType.lower()=="l":
        ab_chain=l

    o_df=pd.read_csv(f'{pops_dir}/{iden_code}-{ab_chain}{antigen}_deltaSASA_rpopsResidue.txt',sep=' ')
    n_df=replace_back_T_F_chain (o_df)

    df=n_df.loc[n_df['D_SASA.A.2']>d_sasa_th] # only keep residues with D_SASA above the threshold
    df=df.loc[df['Chain']==ab_chain] # Only take the antibody chain residues
    df["residue"]=df["ResidNe"].apply(lambda x: seq1(x))

    df["pops_pdb_numbering"]=(df["ResidNr"].astype(str)+df["iCode"]).str.replace("-","")


    df.reset_index(inplace=True)
    return df

def mapping_positions (iden_code,df,seq_col,coor_seq_col,pdb_numbering_col,pops_dir,antigen_col,chainType,d_sasa_th = 15):
    """
    Mapping the positions of the sequence to the positions in coordinate sequence
    args:input:
    iden_code: the identifier code of the antibody in VCAb
    seq_col: Column name for the complete sequence of the chain, i.e. text_seq
    coor_seq_col: Column name for the sequence containing residues with the coordinate
    pdb_numbering_col: Column name for the pdb numbering of the residues with the coordinate, get directory from the VCAb table, in the format of comma separated string
    pops_dir: the directory for the PopsComp results
    antigen_col: Column name for the chain ID of the antigen
    chainType: the chain type of the sequence, either "H" or "L"
    """

    # Get the seq and coor_seq
    seq=df.loc[df["iden_code"]==iden_code,seq_col].values[0]
    coor_seq=df.loc[df["iden_code"]==iden_code,coor_seq_col].values[0]
    pdb_numbering=df.loc[df["iden_code"]==iden_code,pdb_numbering_col].values[0]
    antigen=df.loc[df["iden_code"]==iden_code,antigen_col].values[0]
    if type(antigen)!=str: # if there is no antigen
        return None
    antigen=antigen.split(";")[0]

    # Read the POPS file
    try:
        pops=read_pops_file(iden_code,pops_dir,antigen,chainType, d_sasa_th)
    except:
        pops=None

    # 1. Perform the pairwise alignment
    matrix = substitution_matrices.load('BLOSUM62')
    #matrix = matlist.blosum62
    pair_aln=pairwise2.align.globalds(seq, coor_seq, matrix,-10,-0.5,penalize_end_gaps=False)[0]
    # use blosum62 as the matrix, gap penalties follow the default value of EMBOSS needle
    # just take the first one as the aln result
    #return pair_aln
    alned_num_seq=pair_aln.seqA
    alned_coor_seq=pair_aln.seqB


    # 2. Convert IMGT_num_pos into pure_pos of coor_seq
    # text_num --> pure_pos (text_seq) -->aln_pos (text_seq)=aln_pos(coor_seq) -->pure_pos (coor_seq)
    result={}
    for text_num, text_residue in enumerate(list(seq)):
        text_seq_aln_pos=map_pure_seq_pos_to_aln_pos (text_num,alned_num_seq)
        coor_seq_aln_pos=text_seq_aln_pos

        if coor_seq_aln_pos==None or alned_coor_seq[coor_seq_aln_pos]=="-":
            # if the residue in the alned position of coor_seq is gap
            # this means this residue is missing in the coordinate sequence
            result[text_num]=[text_residue,np.nan,np.nan]
        else:
            # pure_pos for the coor_seq
            coor_seq_pure_pos=map_aln_pos_to_pure_seq_pos(coor_seq_aln_pos,alned_coor_seq)
            this_pdb_numbering=pdb_numbering.split(",")[coor_seq_pure_pos]
            if pops is None:
                if_interface_res=np.nan
            else:
                if_interface_res=len(pops.loc[(pops["residue"]==text_residue)&(pops["pops_pdb_numbering"]==this_pdb_numbering)])
            result[text_num]=[text_residue,this_pdb_numbering,if_interface_res]
    result_df=pd.DataFrame(result,index=["residue","pdb_numbering","if_interface_res"]).T.reset_index().rename(columns={"index":"text_numbering"})
    return result_df

#vcab=pd.read_csv("final_vcab_with_V_coor.csv")

#test_h_result=mapping_positions ("9f1i_IM",vcab,"HV_seq","HV_coordinate_seq","HV_PDB_numbering","pops_result","antigen_chain","H")
#test_l_result=mapping_positions ("9f1i_IM",vcab,"LV_seq","LV_coordinate_seq","LV_PDB_numbering","pops_result","antigen_chain","L")

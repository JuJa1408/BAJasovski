import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from equilibrator_api import ComponentContribution, Q_, Reaction
from scipy import constants 

def getDataBaseCode(metabolite_name):
    #Hashmap to map database IDs of modelSeed, KEGG and Chebi to respective metabolite
    metabolite_name_cpd ={
        'ac': "cpd00029",
        'ATP': "cpd00002",
        'bm': "cpd11416", #(unclear)
        'CO2': "cpd00011",
        'glyc': "cpd00139",
        'glyox': "cpd00040",
        'H2': "cpd11640",
        'H2S': "cpd00239",
        'hv': "cpd31000",
        'NH3': "cpd00013",
        'O2': "cpd00007",
        'PHB': "CHEBI:10983", 
        'polyglc': "C00420",
        'SO4': "cpd00048",

        'ac_CoA' : "cpd00029",
        'akg' : "C00026",
        'ery4p' : "cpd01239",
        'glc6p' : "cpd00027",
        'NADH' : "cpd00004",
        'NH3' : "cpd00013",
        'oaa' : "C00036",
        'PEP' : "cpd00020",
        'pyr' : "C00022",
        'rbo5p' : "cpd00258",

    }
    
    # Find the longest matching key
    longest_match = ""
    for key in metabolite_name_cpd:
        if key in metabolite_name and len(key) > len(longest_match):
            longest_match = key

    # Return the corresponding value if a match is found
    if longest_match:
        return metabolite_name_cpd[longest_match]
    else:
        return None
    
def getBiomass(cc):

    biomass_term ={
    "ac_CoA", 
    "akg",
    "ATP",
    "ery4p",
    "glc6p",
    "NADH", 
    "NH3",
    "oaa",
    "PEP",
    "pyr",
    "rbo5p" 
    }
    biomass_term = list(biomass_term)
    metabolite_num = map(getDataBaseCode, biomass_term)
    compound_list = [cc.get_compound(num) for num in metabolite_num]
    compounds_bm = dict(zip(biomass_term, compound_list))
    standard_dgf_mu, sigmas_fin, sigmas_inf = zip(*map(cc.standard_dg_formation, compound_list))
    standard_dgf_mu = np.array(standard_dgf_mu)
    dgf0_bmterm = pd.DataFrame()
    for name, value in zip(biomass_term, standard_dgf_mu):
        dgf0_bmterm[name] = [value]

    bm = 1.233* dgf0_bmterm['ac_CoA'] + 1.472 * dgf0_bmterm['akg'] + 50 * dgf0_bmterm['ATP'] + 0.531 * dgf0_bmterm['ery4p'] + 0.069 * dgf0_bmterm['glc6p'] + 14.653 * dgf0_bmterm['NADH'] + 12.513 * dgf0_bmterm['NH3'] + 2.379 * dgf0_bmterm['oaa'] + 2.67 * dgf0_bmterm['PEP'] + 4.057 * dgf0_bmterm['pyr'] + 0.787 * dgf0_bmterm['rbo5p']
    return bm, compounds_bm

def getGibbsFreeEnergy(ECM):
    #This function computes the gibbs free energy of each ECM 
    cc = ComponentContribution()
    cc.p_h = Q_(7.5)
    cc.temperature = Q_("298.15K")
    metabolite = list(ECM.columns)
    metabolite_num = map(getDataBaseCode, metabolite)
    
    compound_list = [cc.get_compound(num) for num in metabolite_num]
    compounds = dict(zip(ECM.columns, compound_list))
    standard_dgf_mu, sigmas_fin, sigmas_inf = zip(*map(cc.standard_dg_formation, compound_list))
    standard_dgf_mu = np.array(standard_dgf_mu)
    dgf0 = pd.DataFrame()
    for name, value in zip(metabolite, standard_dgf_mu):
        if 'bm' in name:
            dgf_bm, compounds_bm = getBiomass(cc) 
            dgf0[name] = dgf_bm
        elif 'hv' in name:
            h = (constants.Planck)
            c = float(constants.c)
            avo = float(constants.Avogadro)
            dgf0[name] = ( h * c * 3.79 * 10 ** 14 ) / (1000)   
        else:
            dgf0[name] = [value]        

    
    dgf0 = dgf0.fillna(value=np.nan)
    dgf = ECM.dot(dgf0.transpose())

    return dgf



ECM_comp = pd.read_csv("..\ECMnormalized\ECM_comp_normalized.csv", delimiter=',')
ECM_pool = pd.read_csv("..\ECMnormalized\ECM_pool_normalized.csv", delimiter=',')
ECM_nest = pd.read_csv("..\ECMnormalized\ECM_nest_normalized.csv", delimiter=',')
ECM_fap = pd.read_csv("..\ECMnormalized\ECM_fap_normalized.csv", delimiter=',')
ECM_srb = pd.read_csv("..\ECMnormalized\ECM_srb_normalized.csv", delimiter=',')
ECM_syn = pd.read_csv("..\ECMnormalized\ECM_syn_normalized.csv", delimiter=',')

ECM={
    'comp': ECM_comp,
    'pool': ECM_pool,
    'nest': ECM_nest,
    'fap': ECM_fap,
    'srb': ECM_srb,
    'syn': ECM_syn
}

for model_name, ECM in ECM.items():
    dG = getGibbsFreeEnergy(ECM)
    ECM.insert(len(ECM.columns), 'dG', dG)
    ECM.to_csv('ECM_' + model_name + '_w_dG.csv', index=False)


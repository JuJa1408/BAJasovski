import csv
import pandas as pd
import numpy as np
import scipy.constants
import seaborn as sns
import matplotlib.pyplot as plt
from equilibrator_api import ComponentContribution, Q_, Reaction
from sklearn.preprocessing import StandardScaler
from matplotlib import gridspec
from scipy import constants 
import cobra
from cobra.io import read_sbml_model
import math
import matplotlib.patches as mpatches
import networkx as nx
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError


def getReaction(df):
    """
    This function produces a reaction in form of a string with of a given 
    dataframe (whole ECM matrix or just a vector). Negtive coefficients are 
    substrates and positive coefficients are products. The metabolites are
    the respective column names. 
    """
    metabolites = df.columns[:]

    reactions = {ecm: {'reactants': [], 'products': []} for ecm in df.index} 
    
    for index, row in df.iterrows():
        ecm = row
        for metabolite in metabolites:
            coeff = ecm[metabolite]
            if coeff < 0:
                reactions[index]['reactants'].append(f"{-coeff:.2f} {metabolite}")
            elif coeff > 0:
                reactions[index]['products'].append(f"{coeff:.2f} {metabolite}")

    formatted_reactions = pd.DataFrame(columns=['Reaction'])
    for ecm, sides in reactions.items():
        reaction = ' + '.join(sides['reactants']) + " -> " + ' + '.join(sides['products'])
        
        reaction_df = pd.DataFrame({
            'Reactants': [' + '.join(sides['reactants'])],
            'Arrow': [' -> '],  
            'Products': [' + '.join(sides['products'])]
        })
        
        formatted_reactions.loc[ecm] = reaction
    return formatted_reactions

def dataPreprocessing(path):
    """
    This function processes the date from the produced ECM.csv file from ecmtool 
    to a more readable form. All keyes are removed as well as all virtual_tag 
    columns for this thesis. The original format is still kept and also given back.
    """

    df = pd.read_csv(path, delimiter=',')
    df_unchanged = df.copy()
    
    df.columns = df.columns.str.replace("M_","", regex=True)
    df.columns = df.columns.str.replace("ex","", regex=True)
    df.columns = df.columns.str.replace("fap","", regex=True)
    df.columns = df.columns.str.replace("gen","", regex=False)
    df.columns = df.columns.str.replace("pool","", regex=False)
    df.columns = df.columns.str.replace("srb","", regex=True)
    df.columns = df.columns.str.replace("syn","", regex=False)
    df.columns = df.columns.str.replace("virtual_tag_R_EX_","", regex=True)
    df.columns = df.columns.str.replace("_","", regex=True)

    if 'virtualtagREXac' in df.columns:
        df = df.drop('virtualtagREXac', axis=1)
    for met in df.columns:
        if "virtualtag" in met and met in df.columns:
            df.drop([met], axis=1, inplace=True)
            #df = df.drop(met, axis=0)
    if 'objective' in df.columns:
        df = df.drop('objective', axis=1)    
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    df = df.groupby(level=0, axis=1).sum()
    return df, df_unchanged

def getNormalizedData(ECM):

    """
    This function gives back the normalized form of a dataframe (like an ECM matrix)
    """
    sum_abs_coef = abs(ECM).sum(axis=1)
    normalized_ECM = ECM.div(sum_abs_coef, axis=0)

    return normalized_ECM

def getFBA(model):
    """
    This function computes the FBA of a model. It first makes sure that biomass 
    production is the objective. Biomass in community models are either keyed with _fap_srb_syn
    (that is why we look for the longest string) or have the key _obj. Both for loops are 
    optional as each model has usually an objective set.  
    """
    all_bm_met = []
    for met in model.metabolites:
        if  'bm' in met.id :
            all_bm_met.append(met.id) 
        #print(met)
    bm_met = model.metabolites.get_by_id(max(all_bm_met, key=len))

    for r in model.reactions:
        if bm_met in r.metabolites:
            #r.bounds = (0,1000)
            r.objective_coefficient = 1
            model.objective = r

    optimal = model.optimize()
    if optimal.status == 'optimal':
        other_fluxes = optimal.fluxes

    return optimal, other_fluxes

def helper_model_to_yield_2D(model_name, substrate, product, ECMs):
    """
    This function helps to compute the 2D solution space of an ECM 
    """
    yields = [
        ECMs[model_name][substrate],
        ECMs[model_name][product]
    ]
    return [yields]

def getAnalyzeInteractions(community_fluxes, individual_fluxes):
    """
    This function computes the expected additive effects of each reaction flux with the 
    unicellular models and aferwards it gets compared with the fluxes of the same reactions 
    within a community model. If the flux in the community is higher than the additive effect, 
    there are synergy effects within the community if the flux is below the additive effect 
    there is competition behaviour in the community.
    """
    synergy = {}
    competition = {}
    additive_effects = {}
    common_reactions = set()

    
    common_reactions = set(community_fluxes.index)

    # Bestimme die gemeinsamen Reaktionen
    common_reactions.intersection_update(individual_fluxes.index)
    for reaction in common_reactions:
        community_flux = community_fluxes[reaction]
        if not math.isnan(individual_fluxes[reaction]):
            print(individual_fluxes[reaction])

        individual_sum = np.sum(individual_fluxes[reaction])
        individual_sum = np.sum(individual_sum, axis=0) 

        expected_additive_effect = individual_sum
        additive_effects[reaction] = expected_additive_effect
        
        abs_community_flux = abs(community_flux)
        abs_individual_sum = abs(individual_sum)

        if abs_individual_sum == 0:
            if abs_community_flux == 0:
                synergy[reaction] = 0
                competition[reaction] = 0
                interaction = 0
            else:
                interaction = np.sign(community_flux)*100
        else:
            interaction = (abs_community_flux - abs_individual_sum) / abs_individual_sum * 100

        if interaction > 0:
            synergy[reaction] = interaction
        elif interaction < 0:
            competition[reaction] = interaction

    results_df = pd.DataFrame({'Community Flux': community_fluxes[list(common_reactions)], 
                                'Expected Additive Effect': pd.Series(additive_effects),
                                'Synergy': pd.Series(synergy), 
                                'Competition': pd.Series(competition)})




    return results_df, common_reactions

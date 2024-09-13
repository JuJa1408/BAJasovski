import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cobra
from cobra.io import read_sbml_model
from helper_functions import  *


def getOptimization(ECM_dG, model, method):
    """
    This function first normalizes all ECMs than it sets bimas sproduction as obejctive.
    Afterwards the boundary reactions get mapped to the respective external metabolite of the ECM.
    All bounds of the boundary reactions are first set to 0 and than changed afterwards to the 
    respective ECM coefficient (depending on the optimization method). Within the new bounds 
    the optimization get performed.   
    """
    maxbm = []
    all_bm_met = []
    all_fluxes = {}
    ECM = ECM_dG.drop('dG', axis=1)
    normalized_ECM = getNormalizedData(ECM_dG.drop('dG', axis=1))

    for met in model.metabolites:
        if  'bm' in met.id :
            all_bm_met.append(met.id) 
        #print(met)
    bm_met = model.metabolites.get_by_id(max(all_bm_met, key=len))

    transporters_names = []
    for reac in model.reactions:
        if 'EX' in reac.id or 'DM' in reac.id or 'SK' in reac.id:
            transporters_names.append(reac.id)
    # close all bounds
    for r in model.reactions:
        if r.id.startswith("EX_") or r.id.startswith("DM_") or r.id.startswith("SK_"):
            r.bounds=(0,0)
        if bm_met in r.metabolites:
            r.bounds = (0,1000)
            r.objective_coefficient = 1
            model.objective = r

    transporters = {}
    for transporter in transporters_names: 
        for component in normalized_ECM.columns:
            if component in transporter:
                transporters[component] = transporter    

    for t in transporters_names:
        try:
            r = model.reactions.get_by_id(t)
            print(r)
        except:
            print(f"no transporter {r.id}")

    if method == 1:
        for idx, row in ECM.iterrows():
            for col, val in row.items():
                if val<0:
                    b=(val,0)
                else:
                    b=(0,val)
                if "bm" not in col:
                    model.reactions.get_by_id(transporters[col]).bounds=b
            sol = model.optimize()

            maxbm.append(sol.objective_value)
            if sol.status == 'optimal':
                all_fluxes[idx] = sol.fluxes
            else:
                all_fluxes[idx] = pd.Series([0] * len(model.reactions), index=[r.id for r in model.reactions])

        maxbm_df_vector = pd.DataFrame(index=normalized_ECM.index, data={'maxbm': maxbm, 'dG': ECM_dG['dG']})
        maxbm_df_vector = maxbm_df_vector.sort_values(by='maxbm', ascending=False)
        maxbm_ECMs = ECM.loc[maxbm_df_vector.index[0:5]]
        maxbm_df = np.max(maxbm_df_vector) 
        #sorted_fluxes = pd.DataFrame(all_fluxes).T.loc[maxbm_df.index]
        AllFluxes = pd.DataFrame(index=normalized_ECM.index, data={'maxbm': maxbm, 'otherfluxes': all_fluxes})
        AllFluxes_sorted = AllFluxes.sort_values(by='maxbm', ascending=False)
        optimal_fluxes = AllFluxes_sorted['otherfluxes'].loc[AllFluxes_sorted.index[0]]
    
    elif method == 2:
        # Max Values for every column (for upper bounds)
        max_values = normalized_ECM.max()

        # Min Values for every column (for lower bounds)
        min_values = normalized_ECM.min()
        maxbm_ECMs = 0
        maxbm_df_vector = 0
        # Set bounds for every reaction
        for col, max_val in max_values.items():
            min_val = min_values[col]
            if "bm" not in col:
                reaction = model.reactions.get_by_id(transporters[col])
                if min_val < 0 and max_val > 0:
                    reaction.bounds = (min_val, max_val)
                elif min_val >= 0:
                    reaction.bounds = (0, max_val)
                else:
                    reaction.bounds = (min_val, 0)

        # Optimiie once
        sol = model.optimize()

        if sol.status == 'optimal':
            maxbm_df = sol.objective_value
            optimal_fluxes = sol.fluxes

        else:
            print("Optimization not successful")
    
    return maxbm_df, maxbm_df_vector, maxbm_ECMs, optimal_fluxes

def getFBA(model):
    """
    This old function computes the FBA of a model but it is not used anymore
    """
    all_bm_met = []
    """
    for met in model.metabolites:
        if  'bm' in met.id :
            all_bm_met.append(met.id) 
        #print(met)
    for met in all_bm_met:
        if 'obj' in met:
            bm_met = met
            break
        else:
            bm_met = model.metabolites.get_by_id(max(all_bm_met, key=len))

    for r in model.reactions:
        if bm_met in r.metabolites:
            #r.bounds = (0,1000)
            r.objective_coefficient = 1
            model.objective = r
    """
    optimal = 0
    optimal = model.optimize()
    if optimal.status == 'optimal':
        other_fluxes = optimal.fluxes

    return optimal, other_fluxes

def getSensitivityMatrix(models, nutrient, range_values):
    """
    Input: 
    models = original cobrapy model
    nutrients = some choosen metabolite strings from the respective model (hv, SO4, CO2, NH3, O2, H2) 
    range_values = the column of the choosen metabolites from the ECM of the model 

    Output: 
    Sensitivitymatrix with the nutrients as columns and models as row. 
    Each cell containts the maximal change of objective flux over boundary changes of 
    the respective nutriente.

    This function computes the Sensitivitymatrix (and plots it in form of a heatmap) over all models of some nutrients which shows 
    how much of an impact the boundary change of a metabolite within the model has on the 
    optimization of the objective. It utilizes normalized ECM coefficient as the new boundaries of the model.  
    """
    sensitivitymatrix = pd.DataFrame(index=nutrient, columns=models.keys())
    m=0
    
    for model_name, model in models.items():
        v0 = model.optimize()
        v0 = v0.objective_value / (sum(abs(v0.fluxes)))
        n=0
        for nutrient_id in nutrient:
            for reac in model.reactions:
                if nutrient_id in reac.id:
                    model_nutrient_id = reac.id
            original_upper_bound = model.reactions.get_by_id(model_nutrient_id).upper_bound
            original_lower_bound = model.reactions.get_by_id(model_nutrient_id).lower_bound
            maxdvdx = 0

            for value in range_values[m][n]:
                if value > 0:
                    model.reactions.get_by_id(model_nutrient_id).upper_bound = value
                    dx = value - original_upper_bound
                    vb = model.optimize()
                    vb = vb.objective_value / (sum(abs(vb.fluxes)))
                    dvdx = (vb - v0) / dx   
                elif value < 0:
                    model.reactions.get_by_id(model_nutrient_id).lower_bound = value
                    dx = abs(value) - abs(original_lower_bound)
                    vb = model.optimize()
                    vb = vb.objective_value / (sum(abs(vb.fluxes)))
                    dvdx = (vb - v0) / dx
                else:
                    dvdx = 0
                if abs(dvdx) > maxdvdx:
                    maxdvdx = dvdx
            sensitivitymatrix.at[nutrient_id, model_name] = float(maxdvdx)
            n +=1
            model.reactions.get_by_id(model_nutrient_id).upper_bound = original_upper_bound
            model.reactions.get_by_id(model_nutrient_id).lower_bound = original_lower_bound
        m += 1

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=2)
    sns.heatmap(sensitivitymatrix.astype(float), annot=True, cmap="Blues")
    plt.title('Sensitivitymatrix', fontsize=25)
    plt.xlabel('Model Type', fontsize=23)
    plt.ylabel('Nutrient ID', fontsize=23)
    plt.show()

    return sensitivitymatrix

def plotCorrelationMatrix(maxBiomass, model_name):
    """
    This function plots the correlationmatrix in form of a heatmap 
    from the model "model_name" which shows how much of a correlation between 
    the standard Gibbs free energy of each ECM and the optimized objective flux of each ECM
    """
    sns.set(font_scale=2)
    corr=maxBiomass.corr() 
    x_axis_labels = ['maximal $v_{Biomass}$', '$\Delta G^\circ$']
    y_axis_labels = ['maximal $v_{Biomass}$', '$\Delta G^\circ$']
    sns.heatmap(corr, annot=True, vmax=1, vmin=-1, annot_kws={"size": 20}, xticklabels=x_axis_labels,yticklabels=y_axis_labels)
    plt.title("Correlationmatrix maximal $v_{Biomass}$ vs. $\Delta G^\circ$ for " + model_name + " model")

    return True

def plotOptimizedFluxesFBAvsECM(optimal_FBA, model, ECM_allFlux, model_name):
    """
    This function compares the fluxes of each reaction after both types of 
    optimization with ECM and only FBA. The fluxes get normalized so a better comparison is possible.
    This function plots one lineplot for both set of fluxes   
    """

    FBA_allFlux = optimal_FBA.fluxes
    FBA_allFlux_norm = FBA_allFlux/(sum(abs(FBA_allFlux)))
    ECM_allFlux_norm = ECM_allFlux/(sum(abs(ECM_allFlux)))

    fig1, ax = plt.subplots(figsize=(12, 8))
    fig1.suptitle("Flux Comparison for " + model_name + " model" , fontsize=30)
    ax.plot(ECM_allFlux_norm.index, ECM_allFlux_norm, label='ECM optimization ', color='blue', linestyle='--', marker='x')
    ax.plot(FBA_allFlux_norm.index, FBA_allFlux_norm, label = 'FBA optimization', color='red', linestyle='--', marker='x')
    ax.set_ylabel('Normalized Flux Value', fontsize=25)
    ax.set_xlabel('Reactions', fontsize=25)
    plt.xticks(rotation=90, fontsize=18)
    plt.axhline(0, color='grey', linewidth=0.5)
    plt.grid(True)
    plt.legend(fontsize=27.5)
    #plt.text(0, -0.15 , f"ECM: {getReaction(maxBiomassECM_pool).iloc[0,0]}", fontsize=16, ha='left', wrap=True, bbox=dict(facecolor='white', pad=10.0))
    plt.tight_layout()
    plt.show()
    return True

def plotOptimizedFluxvsdG(maxBiomass, model_name):
    """
    This function plots the optimized objective flux of the model "model_name" in a sorted barplot.
    Each column/bar represents the flux of one ECM. Every column has also a cross x which shows the 
    Gibbs free energy of the respective ECM. 
    """
    Sorted_maxBiomass = maxBiomass[maxBiomass['maxbm']>0].sort_values(by='maxbm', ascending=False).reset_index(drop=True)
    fig, ax1 = plt.subplots()
    bars = ax1.bar(Sorted_maxBiomass.index, Sorted_maxBiomass['maxbm'], color='tab:blue', label='Maximal Flux vBM')
    ax2 = ax1.twinx()
    ax2.scatter(Sorted_maxBiomass.index, Sorted_maxBiomass['dG'], marker='x', color='tab:red', s=180, linewidths=3, label='Gibbs free energy')
    ax1.set_xlabel('ECMs', fontsize=20)
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_ylabel('Maximal Flux for Biomassproduction', color='tab:blue', fontsize=20)
    ax2.set_ylabel('Gibbs free energy [kJ*mol^-1] of respective ECM', color='tab:red', fontsize=20)
    plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    plt.title('Maximal Flux for each ECM with respective Gibbs Free Energy (' + model_name + ')', fontsize=25)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower left', fontsize=22)
    plt.show()
    return True

def helperInteractionAnalysis(community_fluxes, individual_fluxes):
    """
    This function computes the interaction metrics of the community model compared to the
    additive effects of the individual models. First it finds the common reactions of the community model
    compared to the additive unicellular models. Than the interaction gets quatified as synergy 
    (with the interaction value being >0) and competition (with the interaction value being <0).
    """
    synergy = {}
    competition = {}
    additive_effects = {}
    
    common_reactions = set(community_fluxes.index)

    # Find common reactions
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

def getInteractionAnalysis(allFlux_fap, allFlux_srb, allFlux_syn, allFlux_comp, allFlux_pool):
    """
    This function computes the interaction analysis of multiple community models compared to 
    the additive effects of the unicellular models. The fluxes that are looked at are normalized 
    first for a better comparisson between community models.
    """
    allFlux_additive = pd.concat([allFlux_fap, allFlux_srb, allFlux_syn], axis=0)
    sumFlux = (sum(abs(allFlux_additive)))
    allFlux_additive = allFlux_additive/sumFlux

    allFlux_comp = allFlux_comp/(sum(abs(allFlux_comp)))
    allFlux_pool =allFlux_pool/(sum(abs(allFlux_pool)))
   
    individual_fluxes = allFlux_additive#pd.concat([allFlux_fap, allFlux_srb, allFlux_syn], axis=0)

    compAnalysis, commonreactions = helperInteractionAnalysis(allFlux_comp, individual_fluxes)
    plot_synergy_competition(compAnalysis, 'Compartmentalized')

    poolAnalysis, commonreactions = helperInteractionAnalysis(allFlux_pool, individual_fluxes)
    plot_synergy_competition(poolAnalysis, 'Pooled')

    combined_df = pd.DataFrame({
        'Community Model 1': compAnalysis['Community Flux'],
        'Community Model 2': poolAnalysis['Community Flux'],
        'Expected Additive Effect': compAnalysis['Expected Additive Effect']
    })
    sorted_df = combined_df.reindex(commonreactions).sort_values(by='Expected Additive Effect')

    plot_combined_synergy_competition_line(sorted_df.index, combined_df, sorted_df['Expected Additive Effect'].values)

    return True

def plot_synergy_competition(results_df, modelname):
    """
    This function shows for one model the interaction value of each reaction in form of a bar plot.
    positive interaction values are blue and represent synergies within the community 
    and negative interaction values are orange and represent the competition within the community.
    """
    # Combine synergy and competition in one column
    results_df['Effect'] = results_df['Synergy'].combine_first(results_df['Competition'])
    
    # Prepare data fro plotting
    effects = results_df['Effect']
    colors = ['blue' if val > 0 else 'orange' for val in effects]
    maxcompetionEffect = np.min(effects)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(effects.index, effects, color=colors, label='Synergy/Competition')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Synergy and Competition in ' + modelname + ' Model', fontsize=22)
    synergy_patch = mpatches.Patch(color='blue', label='Synergy (>0)')
    competition_patch = mpatches.Patch(color='orange', label='Competition (<0)')
    ax.legend(handles=[synergy_patch, competition_patch], fontsize=18)
    plt.ylim(maxcompetionEffect, abs(maxcompetionEffect))
    plt.xticks(rotation=90)
    plt.show()
    return True

def plot_combined_synergy_competition_line(common_reactions, combined_df, sorted_expected_additive_effect):
    """
    This function plot all interaction values of each reaction of the common reaction set to 
    compare the expected additive fluxes with the community models in one lineplot  
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot expected additiven effects
    ax.plot(common_reactions, sorted_expected_additive_effect, label='Expected Additive Effect', color='black', linestyle='-', marker='o')

    # Plot fluxes of community model
    ax.plot(common_reactions, combined_df['Community Model 1'].reindex(common_reactions).values, label='Compartmentalized Community Model', color='blue', linestyle='--', marker='x')
    ax.plot(common_reactions, combined_df['Community Model 2'].reindex(common_reactions).values, label='Pooled Community Model', color='orange', linestyle='--', marker='x')

    ax.set_ylabel('Flux Value', fontsize=20)
    ax.set_xlabel('Common Reactions', fontsize=20)
    ax.set_title('Synergy and Competition in Microbial Communities', fontsize=25)
    plt.xticks(rotation=45, fontsize=18)
    plt.axhline(0, color='grey', linewidth=0.5)
    plt.grid(True)
    plt.legend(fontsize=22)
    plt.show()
    return True

ECM_comp = pd.read_csv("..\ECMwithdG\ECM_comp_w_dG.csv", delimiter=',')
ECM_pool = pd.read_csv("..\ECMwithdG\ECM_pooled_w_dG.csv", delimiter=',')
ECM_nest = pd.read_csv("..\ECMwithdG\ECM_nested_w_dG.csv", delimiter=',')
ECM_fap = pd.read_csv("..\ECMwithdG\ECM_fap_w_dG.csv", delimiter=',')
ECM_syn = pd.read_csv("..\ECMwithdG\ECM_syn_w_dG.csv", delimiter=',')
ECM_srb = pd.read_csv("..\ECMwithdG\ECM_srb_w_dG.csv", delimiter=',')

ECM={
    'comp': ECM_comp,
    'pool': ECM_pool,
    'nest': ECM_nest,
    'fap': ECM_fap,
    'srb': ECM_srb,
    'syn': ECM_syn
}

model={
    'comp': cobra.io.read_sbml_model('..\models\compartmentalized_model.xml'),
    'pool': cobra.io.read_sbml_model('..\models\pooled_model.xml'),
    'nest': cobra.io.read_sbml_model('..\models\\nested_model.xml'),
    'fap': cobra.io.read_sbml_model('..\models\\fap.xml'),
    'srb': cobra.io.read_sbml_model('..\models\srb.xml'),
    'syn': cobra.io.read_sbml_model('..\models\syn.xml')
}
optimal={}

optimal['srb'] = model['srb'].optimize()
optimal['comp'] = model['comp'].optimize()
optimal['pool'] = model['pool'].optimize()
optimal['nest'] = model['nest'].optimize()
optimal['fap'] = model['fap'].optimize()
optimal['syn'] = model['syn'].optimize()

communitymodel={
    'comp': cobra.io.read_sbml_model('..\models\compartmentalized_model.xml'),
    'pool': cobra.io.read_sbml_model('..\models\pooled_model.xml'),
    'nest': cobra.io.read_sbml_model('..\models\\nested_model.xml')
}

maxbmflux={}
maxbm_vector={}
maxbm_ECMs={}
optimal_fluxes={}

for model_name, ECM in ECM.items():
    maxbmflux[model_name], maxbm_vector[model_name], maxbm_ECMs[model_name], optimal_fluxes[model_name] = getOptimization(ECM ,model[model_name],1)
    plotCorrelationMatrix(maxbm_vector[model_name], model_name)
    plotOptimizedFluxesFBAvsECM(optimal[model_name], model[model_name], optimal_fluxes[model_name], model_name)
    plotOptimizedFluxvsdG(maxbm_vector[model_name],model_name)

#The following function computes and plots the interaction analysis of comp and pool compared to the expected additive effects
"""
getInteractionAnalysis(optimal_fluxes["fap"], optimal_fluxes['srb'], optimal_fluxes['syn'], optimal_fluxes['comp'], optimal_fluxes['pool'])
"""

#The following section computes the sensitivity analysis of all community model types
"""
ECM_comp_norm = getNormalizedData(ECM_comp)
ECM_pool_norm = getNormalizedData(ECM_pool)
ECM_nest_norm = getNormalizedData(ECM_nest)

nutrients =["hv", "SO4", "CO2", "NH3", "O2", "H2"]

range_values = [[ECM_comp_norm['hv_gen'], 
                 ECM_comp_norm['SO4ex_gen'], 
                 ECM_comp_norm['CO2ex_gen'], 
                 ECM_comp_norm['NH3_syn'], 
                 ECM_comp_norm["O2expool_gen"], 
                 ECM_comp_norm['H2ex_gen']],

                [ECM_pool_norm['hv'],
                 ECM_pool_norm['SO4'],
                 ECM_pool_norm['CO2'], 
                 ECM_pool_norm['NH3'], 
                 ECM_pool_norm['O2'], 
                 ECM_pool_norm['H2']],

                [ECM_nest_norm['hv_gen_fap'], 
                 ECM_nest_norm['SO4ex_gen_srb'], 
                 ECM_nest_norm['CO2ex_gen_fap'], 
                 ECM_nest_norm['NH3_syn'], 
                 ECM_nest_norm['O2expool_gen_fap'], 
                 ECM_nest_norm['H2pool_gen_fap']]]

SensitivityMatrix = getSensitivityMatrix(communitymodel, nutrients, range_values)
"""
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cobra
from cobra.io import read_sbml_model
from helper_functions import  *


def getOptimization(ECM_dG, model, method):
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

    #transporters = dict(zip(ECM.columns, transporters_names))
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
            #print(sol.objective_value)
            #print("END")
            maxbm.append(sol.objective_value)
            if sol.status == 'optimal':
                all_fluxes[idx] = sol.fluxes#pd.Series(data={'fluxes': sol.fluxes}, index=[r.id for r in m.reactions] )
            else:
                all_fluxes[idx] = pd.Series([0] * len(m.reactions), index=[r.id for r in m.reactions])

        maxbm_df_vector = pd.DataFrame(index=normalized_ECM.index, data={'maxbm': maxbm, 'dG': ECM_dG['dG']})
        maxbm_df_vector = maxbm_df_vector.sort_values(by='maxbm', ascending=False)
        maxbm_ECMs = ECM.loc[maxbm_df_vector.index[0:5]]
        maxbm_df = np.max(maxbm_df_vector) 
        #sorted_fluxes = pd.DataFrame(all_fluxes).T.loc[maxbm_df.index]
        AllFluxes = pd.DataFrame(index=normalized_ECM.index, data={'maxbm': maxbm, 'otherfluxes': all_fluxes})
        AllFluxes_sorted = AllFluxes.sort_values(by='maxbm', ascending=False)
        optimal_fluxes = AllFluxes_sorted['otherfluxes'].loc[AllFluxes_sorted.index[0]]
    
    elif method == 2:
        # Maximale Werte für jede Spalte finden (für upper bounds)
        max_values = normalized_ECM.max()

        # Minimale Werte für jede Spalte finden (für lower bounds)
        min_values = normalized_ECM.min()
        maxbm_ECMs = 0
        maxbm_df_vector = 0
        # Setze die Grenzen für jede Reaktion
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

        # Optimiere einmal
        sol = model.optimize()

        if sol.status == 'optimal':
            maxbm_df = sol.objective_value
            optimal_fluxes = sol.fluxes

        else:
            print("Optimization not successful")
    
    return maxbm_df, maxbm_df_vector, maxbm_ECMs, optimal_fluxes

def getFBA(model):
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

def getSensitivityMatrix(models, nutrient, range_values):
    
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
    sns.set(font_scale=2)
    corr=maxBiomass.corr() 
    x_axis_labels = ['maximal $v_{Biomass}$', '$\Delta G^\circ$']
    y_axis_labels = ['maximal $v_{Biomass}$', '$\Delta G^\circ$']
    sns.heatmap(corr, annot=True, vmax=1, vmin=-1, annot_kws={"size": 20}, xticklabels=x_axis_labels,yticklabels=y_axis_labels)
    plt.title("Correlationmatrix maximal $v_{Biomass}$ vs. $\Delta G^\circ$ for " + model_name + " model")

    return True

def plotOptimizedFluxesFBAvsECM(model, ECM_allFlux, model_name):

    objectiveFlux, FBA_allFlux = getFBA(model)
    FBA_allFlux_norm = FBA_allFlux/(sum(abs(FBA_allFlux)))
    ECM_allFlux_norm = ECM_allFlux/(sum(abs(ECM_allFlux)))

    fig1, ax = plt.subplots(figsize=(12, 8))
    fig1.suptitle("Flux Comparison for " + model_name + " model" , fontsize=30)
    ax.plot(ECM_allFlux_norm.index, ECM_allFlux_norm, label='ECM optimization ', color='blue', linestyle='--', marker='x')
    ax.plot(FBA_allFlux_norm.index, FBA_allFlux_norm, label = 'FBA optimization', color='red', linestyle='--', marker='x')
    ax.set_ylabel('Normalized Flux Value', fontsize=25)
    ax.set_xlabel('Reactions', fontsize=25)
    #plt.ylim(-3000,3000)
    plt.xticks(rotation=90, fontsize=18)
    plt.axhline(0, color='grey', linewidth=0.5)
    plt.grid(True)
    plt.legend(fontsize=27.5)
    #plt.title( fontsize=25)
    #plt.text(0, -0.15 , f"ECM: {getReaction(maxBiomassECM_pool).iloc[0,0]}", fontsize=16, ha='left', wrap=True, bbox=dict(facecolor='white', pad=10.0))
    plt.tight_layout()
    plt.show()
    return True

def plotOptimizedFluxvsdG(maxBiomass, model_name):
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
    plotOptimizedFluxesFBAvsECM(model[model_name], optimal_fluxes[model_name], model_name)
    plotOptimizedFluxvsdG(maxbm_vector[model_name],model_name)


#The following section computes the sensitivity analysis of all community model types

#ECM_comp_norm = getNormalizedData(ECM_comp)
#ECM_pool_norm = getNormalizedData(ECM_pool)
#ECM_nest_norm = getNormalizedData(ECM_nest)

#nutrients =["hv", "SO4", "CO2", "NH3", "O2", "H2"]
#range_values = [[ECM_comp_norm['hv_gen'], 
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

#SensitivityMatrix = getSensitivityMatrix(communitymodel, nutrients, range_values)

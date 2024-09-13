import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from cobra.io import read_sbml_model
from helper_functions import  *

def plotECM(ECM):
    """
     This function plots the heatmap of an ECM with a blue cell representing a substrate 
     and a red cell representing a product. Right now the plot is set with only signs of the coefficient but this can be changed 
     when removing the np.sign() and optionally adding vmax=10 and vmin=-10  
    
    """

    fig1 = plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])  

    ax0 = fig1.add_subplot(gs[0])
    sns.heatmap(np.sign(ECM.transpose()), cmap='coolwarm', ax=ax0, center=0, annot_kws={'size': 30})
    ax0.set_title('ECMs of the nested model with signs', fontsize= 25)
    ax0.tick_params(axis='y', labelsize=18)
    ax0.tick_params(axis='x', labelsize=18)
    ax0.set_xlabel('ECMs', fontsize=23)
    #if dgf_bool:
    #    ax1 = fig1.add_subplot(gs[1])
    #    sns.heatmap(dgf.transpose(), annot=False, yticklabels=False, cmap='PRGn', ax=ax1, center=0,vmax=10000, vmin=-10000)
    #    ax1.set_title('Gibbs free energy of reaction for each ECM [kJ/mol]')
    plt.tight_layout()
    plt.show()
    return True

def plotSorteddG(ECM, dgf):
    """
    This function plots the Gibbs free energy of every ECM and sorts them in  a barplot
    the comments inbetween lines are to access the 3 ECMs with highest Gibbs free energy 
    and the 3 with lowest Gibbs free energy 

    """

    positive_dgf = (ECM['dG'] >= 0).sum()
    print(ECM.loc[ECM['dG'] >= 0])
    print(positive_dgf)
    #positive_ECMs= getReaction(ECM_fap.loc[ECM_fap['dG'] >= 0])
    #print(positive_ECMs)

    sorted_dgf = dgf.sort_values(ascending=True)
    smallest_dg_indices = sorted_dgf.head(3).index
    biggest_dg_indices = sorted_dgf.tail(3).index
    ECM = ECM.drop('dG', axis=1)
    #smallest_dgf_ECMs = ECM.loc[smallest_dg_indices]
    #biggest_dgf_ECMs = ECM.loc[biggest_dg_indices]


    #reaction_strings_smallest = getReaction(smallest_dgf_ECMs)
    #reaction_strings_biggest = getReaction(biggest_dgf_ECMs)


    sorted_dgf.plot(kind='bar', legend=False)
    first_positive_index = sorted_dgf[sorted_dgf > 0].index[0]

    line_position = sorted_dgf.index.get_loc(first_positive_index)

    plt.axvline(x=line_position - 0.5, color='orange', linestyle='--',
            label='Boundary between negative and positive', linewidth=4)

    #plt.ylim(-20000, 5000)
    #plt.yscale('log')
    plt.xticks([], fontsize=18)
    plt.xlabel('ECMs', fontsize=20)
    plt.ylabel('standard Gibbs free energy [kJ*mol^-1]', fontsize=20)
    plt.title('Gibbs free energy for every ECM of fap model', fontsize=25)

    # Add text annotations for the reactions below the plot
    #for i, (idx, reaction) in enumerate(reaction_strings_biggest.iterrows()):
     #   plt.text(-400 , 4700 - (800*(i+2)), f"ECM {idx}: {reaction['Reaction']}", fontsize=10, ha='left')

    #for i, (idx, reaction) in enumerate(reaction_strings_smallest.iterrows()):
     #   plt.text(-400, -20000 - (800*(i+2)), f"ECM {idx}: {reaction['Reaction']}", fontsize=10, ha='left')

    plt.show()
    return True


    return True

def plotCompareShareMetabolitesInAndOut(Cells):
    """
    This function plots the first interaction analysis heatmap where on the left side there 
    are the perecentage of accurance of a metabolite over all ECMs from one microbe and on 
    the right side the respective percentage of product occurance 

    """
    metabolites = set()
    for df in Cells.values():
        metabolites.update(df.columns)
    metabolites = list(metabolites)
    

    IO = ["Product", "Substrate"]
    AbsRel = ["Coeff", "Percentage"]

    multi_index_columns = pd.MultiIndex.from_product([IO,Cells,AbsRel], names=['InOut', 'Cell', 'AbsRel'])

    df = pd.DataFrame(data=np.zeros((len(metabolites), len(multi_index_columns))),
                      index=metabolites, 
                      columns=multi_index_columns)


    for cell_name, cell_df in Cells.items():
        for metabolite in metabolites:
            countInput = 0
            countOutput = 0
            coeffsumInput = 0
            coeffsumOutput = 0
            for index, row in cell_df.iterrows():
                OutCellEcm = index
                coeff = row.get(metabolite,0)
                if coeff < 0:
                    coeffsumInput += coeff
                    countInput +=1
                elif coeff > 0:
                    coeffsumOutput += coeff
                    countOutput +=1
            if countOutput != 0:
                df.at[metabolite, ("Product", cell_name, "Coeff")] = coeffsumOutput / (countInput+countOutput)
            if countInput != 0:
                df.at[metabolite, ("Substrate", cell_name, "Coeff")] = coeffsumInput /  (countInput+countOutput)
            df.at[metabolite, ("Product", cell_name, "Percentage")] = countOutput / (OutCellEcm+1) 
            df.at[metabolite, ("Substrate", cell_name, "Percentage")] = countInput / (OutCellEcm+1)
            
    outMatrix = df.round(2)
    coeff_data = outMatrix.xs('Coeff', level='AbsRel', axis=1)
    coeff_data = coeff_data.round(2)
    percentage_data = outMatrix.xs('Percentage', level='AbsRel', axis=1)
    percentage_data = percentage_data.round(2) * 100
    percentage_data["Substrate"] = percentage_data["Substrate"] * -1

    coeff_data.columns = ["fap","srb", "syn", "fap", "srb", "syn"]
    percentage_data.columns = ["fap","srb", "syn", "fap", "srb", "syn"]

    annotations = [[f"{p}%" for c, p in zip(coeff_row, perc_row)]
                for coeff_row, perc_row in zip(coeff_data.values.round(2), percentage_data.values.astype(int))]

    assert coeff_data.shape == (len(annotations), len(annotations[0]))
    sns.set(style='white', font_scale=1.4)
    plt.figure(figsize=(10, 8))
    mask = percentage_data == 0
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    print(annotations)
    ax = sns.heatmap(percentage_data, cmap=cmap, center=0, vmin=-100, vmax=100, mask=mask, annot=annotations, fmt="s",
                    linewidths=0.5, linecolor='white')
    ax.xaxis.tick_top()
    ax.set_title('Share of occurence in all ECMs of each microbe', fontsize= 25)
    cmap.set_bad('white')  
    ax.vlines([3],*ax.get_ylim(),colors="black")

    plt.show()

    return True

def plotSynergyCompetitionLine(common_reactions, combined_df, sorted_expected_additive_effect):
    """
    This plot generates the interaction analysis in form of a line plot. Here it first computes
    the expected additive effects of each reaction flux with the unicellular models and aferwards 
    it gets compared with the fluxes of the same reactions within a community model. If the flux in 
    the community is higher than the additive effect, there are synergy effects within the community 
    if the flux is below the additive effect there is competition behaviour in the community.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot of expected additive effects
    ax.plot(common_reactions, sorted_expected_additive_effect, label='Expected Additive Effect', color='black', linestyle='-', marker='o')

    # Plot flux of community model
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


ECM_comp_norm, ECM_comp_unchanged = dataPreprocessing("..\ECMnormalized\ECM_comp_normalized.csv")
ECM_pool_norm, ECM_pool_unchanged = dataPreprocessing("..\ECMnormalized\ECM_pool_normalized.csv")
ECM_nest_norm, ECM_nest_unchanged = dataPreprocessing("..\ECMnormalized\ECM_nest_normalized.csv")
ECM_fap_norm, ECM_fap_unchanged = dataPreprocessing("..\ECMnormalized\ECM_fap_normalized.csv")
ECM_srb_norm, ECM_srb_unchanged = dataPreprocessing("..\ECMnormalized\ECM_srb_normalized.csv")
ECM_syn_norm, ECM_syn_unchanged = dataPreprocessing("..\ECMnormalized\ECM_syn_normalized.csv")

ECM={
    'fap': ECM_fap,
    'srb': ECM_srb,
    'syn': ECM_syn
}

plotSorteddG(ECM_fap, ECM_fap['dG'])

#plotCompareShareMetabolitesInAndOut(ECM)

print("Done")

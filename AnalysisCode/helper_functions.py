import pandas as pd


def getReaction(df):

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

    sum_abs_coef = abs(ECM).sum(axis=1)
    normalized_ECM = ECM.div(sum_abs_coef, axis=0)

    return normalized_ECM

def helper_model_to_yield_2D(model_name, substrate, product, ECMs):
    yields = [
        ECMs[model_name][substrate],
        ECMs[model_name][product]
    ]
    return [yields]



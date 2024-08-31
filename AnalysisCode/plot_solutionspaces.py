import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cobra.io import read_sbml_model
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from helper_functions import  *



def helper_model_to_yield_2D(model_name, substrate, product, ECMs):
    # Diese Funktion simuliert die Berechnung von ECM-Yields
    # In einer echten Anwendung wird hier der ECM-Algorithmus aufgerufen    
    yields = [
        ECMs[model_name][substrate],
        ECMs[model_name][product]
    ]
    return [yields]

def compare_ecm_solution_spaces(substrate, product, ECMs):
    colors = ['b', 'r', 'g']
    plt.figure()
    i=1
    for model_name, ECM in ECMs.items():
        # Berechne die ECMs für das aktuelle Modell
        list_of_yield = helper_model_to_yield_2D(model_name, substrate, product, ECMs)
        list_row = list_of_yield[0]
        
        # Zeichne die ECM-Punkte
        plt.scatter(list_row[0], list_row[1], label=model_name, color=colors[i % len(colors)])
        
        # Berechne und zeichne die konvexe Hülle, falls genügend Punkte vorhanden sind
        if len(list_row[0]) >= 3:
            points = np.array(list_row).T
            try:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], colors[i % len(colors)] + '-', linewidth=3)
            except QhullError as e:
                print(f"QhullError for model {model_name}: {e}")
                # Joggling the points slightly to avoid coplanarity issues
                points += 1e-9 * np.random.rand(*points.shape)
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], colors[i % len(colors)] + '-', linewidth=3 )
        i+=1
    plt.rcParams.update({'font.size': 25})
    plt.xlabel(substrate, fontsize=25)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylabel(product, fontsize=25)
    plt.legend(fontsize=22)
    plt.title(f'Case study {substrate} to {product}', fontsize=27)
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.show()


ECM_comp_norm = pd.read_csv("..\ECMnormalized\ECM_comp_normalized.csv", delimiter=',')
ECM_pool_norm = pd.read_csv("..\ECMnormalized\ECM_pool_normalized.csv", delimiter=',')
ECM_nest_norm = pd.read_csv("..\ECMnormalized\ECM_nest_normalized.csv", delimiter=',')


ECM_A = pd.DataFrame({
            "hv": ECM_comp_norm['hv_gen'],
            'biomass': ECM_comp_norm['bm_fap_srb_syn']
            }
        )
ECM_B = pd.DataFrame({
            'hv': ECM_pool_norm['hv'],
            'biomass': ECM_pool_norm['bm']
            }
        )
ECM_C = pd.DataFrame({
            'hv': ECM_nest_norm['hv_gen_syn'],
            'biomass': ECM_nest_norm['bm_fap_srb_syn']
            }
        )
ECMs = {
    "Compartmentalized": ECM_A,
    "Pooled": ECM_B,
    "Nested": ECM_C
}
compare_ecm_solution_spaces("hv", "biomass", ECMs)
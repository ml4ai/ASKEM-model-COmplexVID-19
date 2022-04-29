# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:44:23 2020

<lang:english>
Main code to perform an experiment. Run in parallel using the
     number of colors - 2, each thread is an iteration, final results are the average
     Here it is fixed at 10 iterations, in the article we used 100 (ideal, but expensive)
     The script has several parameters that must be set manually, the
     comments should guide you. Any questions please contact:
         Leonardo Scabini, scabini@ifsc.usp.br
</lang:english>

<lang:portuguese>
Codigo principal para realizar um experimento. Roda em paralelo usando o
    numero de cores - 2, cada thread é uma iteração, resultados finais sao a media
    Aqui esta fixo em 10 iterações, no artigo usamos 100 (ideal, mas custoso) 
    O script tem varios parametros que devem ser setados manualmente, os 
    comentários devem guiá-lo. Qualquer dúvida entre em contato com:
        Leonardo Scabini, scabini@ifsc.usp.br
</lang:portuguese>

@author: scabini
"""

import os
import sys
from multiprocessing import Pool
import networkx as nx
import model 
from model import isolate_node
import numpy as np
import random
from collections import Counter
from comunities import *
# from dynamics import *
from dynamics_OTM import *
import random
import matplotlib.pyplot as plt
import pickle
from time import perf_counter 
import matplotlib.ticker as ticker

age_dist = np.zeros((6))
fam_structure = np.zeros((10))

####################### PARAMETERS #################################################################################
cidade = "Sao carlos"
n = 251983  # city population
# we test with n up to 250k, if the population is much larger than that, simulate
# with n=100k and change the scale factor
populacao_br = n  # scale factor to approximate the results according to
# the actual population of the study site *(see paper)

beta_global = 0.3  # article default = 0.3, simulates reduction of contagion rate in existing activities

# age distribution, sum must be 1
age_dist[0] = 0.18  # 0-13  - schools
age_dist[1] = 0.06  # 14-17 - schools
age_dist[2] = 0.11  # 18-24 - company
age_dist[3] = 0.23  # 25-39 - company
age_dist[4] = 0.26  # 40-59 - company
age_dist[5] = 0.16  # 60+

# family size distribution, sum must be 1
fam_structure[0] = 0.12  # 1 people -
fam_structure[1] = 0.22  # 2 people
fam_structure[2] = 0.25  # 3 people
fam_structure[3] = 0.21  # 4 people
fam_structure[4] = 0.11  # 5 people-
fam_structure[5] = 0.05  # 6 people-
fam_structure[6] = 0.02  # 7 people-
fam_structure[7] = 0.01  # 8 people-
fam_structure[8] = 0.006  # 9 people-
fam_structure[9] = 0.004  # 10 people-

qtde_religiao = 0.4  # fraction of the population that goes to church once a week
qtde_transporte = 0.36  # fraction of the population that uses public transport
tempo_transporte = 1.2   # time per day in hours, spent on transport
primeiro_infectado = "Março 18"
layers_0 = ['casas', 'aleatorio', 'trabalho', 'transporte', 'escolas', 'igrejas']  # initial layers of the model, in the case of starting in the SP quarantine, 4
acoes = ["Março 24", "Maio 25"]  # vector with the days of measurements, ex 24/3 of SP
layers_tirar = [["transporte", "escolas", "igrejas"], ["trabalho"]]  # list of lists, layers to remove in each action
layers_por = [[], []]  # lists of lists, layers to add on each action
####################################################################################################################


repetitions = 10  # simulation repetitions, final results are the average. In the paper we did 100 repetitions (costly)

layer_names = ['casas', 'aleatorio', 'trabalho', 'transporte', 'escolas', 'igrejas']

# facilities, list with the days of the year (this format can be simulated until December of this year)
ano = {'Janeiro 1': 0, 'Janeiro 2': 1, 'Janeiro 3': 2, 'Janeiro 4': 3, 'Janeiro 5': 4, 'Janeiro 6': 5, 'Janeiro 7': 6, 'Janeiro 8': 7, 'Janeiro 9': 8, 'Janeiro 10': 9, 'Janeiro 11': 10, 'Janeiro 12': 11, 'Janeiro 13': 12, 'Janeiro 14': 13, 'Janeiro 15': 14, 'Janeiro 16': 15, 'Janeiro 17': 16, 'Janeiro 18': 17, 'Janeiro 19': 18, 'Janeiro 20': 19, 'Janeiro 21': 20, 'Janeiro 22': 21, 'Janeiro 23': 22, 'Janeiro 24': 23, 'Janeiro 25': 24, 'Janeiro 26': 25, 'Janeiro 27': 26, 'Janeiro 28': 27, 'Janeiro 29': 28, 'Janeiro 30': 29, 'Janeiro 31': 30, 'Fevereiro 1': 31, 'Fevereiro 2': 32, 'Fevereiro 3': 33, 'Fevereiro 4': 34, 'Fevereiro 5': 35, 'Fevereiro 6': 36, 'Fevereiro 7': 37, 'Fevereiro 8': 38, 'Fevereiro 9': 39, 'Fevereiro 10': 40, 'Fevereiro 11': 41, 'Fevereiro 12': 42, 'Fevereiro 13': 43, 'Fevereiro 14': 44, 'Fevereiro 15': 45, 'Fevereiro 16': 46, 'Fevereiro 17': 47, 'Fevereiro 18': 48, 'Fevereiro 19': 49, 'Fevereiro 20': 50, 'Fevereiro 21': 51, 'Fevereiro 22': 52, 'Fevereiro 23': 53, 'Fevereiro 24': 54, 'Fevereiro 25': 55, 'Fevereiro 26': 56, 'Fevereiro 27': 57, 'Fevereiro 28': 58, 'Fevereiro 29': 59, 'Março 1': 60, 'Março 2': 61, 'Março 3': 62, 'Março 4': 63, 'Março 5': 64, 'Março 6': 65, 'Março 7': 66, 'Março 8': 67, 'Março 9': 68, 'Março 10': 69, 'Março 11': 70, 'Março 12': 71, 'Março 13': 72, 'Março 14': 73, 'Março 15': 74, 'Março 16': 75, 'Março 17': 76, 'Março 18': 77, 'Março 19': 78, 'Março 20': 79, 'Março 21': 80, 'Março 22': 81, 'Março 23': 82, 'Março 24': 83, 'Março 25': 84, 'Março 26': 85, 'Março 27': 86, 'Março 28': 87, 'Março 29': 88, 'Março 30': 89, 'Março 31': 90, 'Abril 1': 91, 'Abril 2': 92, 'Abril 3': 93, 'Abril 4': 94, 'Abril 5': 95, 'Abril 6': 96, 'Abril 7': 97, 'Abril 8': 98, 'Abril 9': 99, 'Abril 10': 100, 'Abril 11': 101, 'Abril 12': 102, 'Abril 13': 103, 'Abril 14': 104, 'Abril 15': 105, 'Abril 16': 106, 'Abril 17': 107, 'Abril 18': 108, 'Abril 19': 109, 'Abril 20': 110, 'Abril 21': 111, 'Abril 22': 112, 'Abril 23': 113, 'Abril 24': 114, 'Abril 25': 115, 'Abril 26': 116, 'Abril 27': 117, 'Abril 28': 118, 'Abril 29': 119, 'Abril 30': 120, 'Maio 1': 121, 'Maio 2': 122, 'Maio 3': 123, 'Maio 4': 124, 'Maio 5': 125, 'Maio 6': 126, 'Maio 7': 127, 'Maio 8': 128, 'Maio 9': 129, 'Maio 10': 130, 'Maio 11': 131, 'Maio 12': 132, 'Maio 13': 133, 'Maio 14': 134, 'Maio 15': 135, 'Maio 16': 136, 'Maio 17': 137, 'Maio 18': 138, 'Maio 19': 139, 'Maio 20': 140, 'Maio 21': 141, 'Maio 22': 142, 'Maio 23': 143, 'Maio 24': 144, 'Maio 25': 145, 'Maio 26': 146, 'Maio 27': 147, 'Maio 28': 148, 'Maio 29': 149, 'Maio 30': 150, 'Maio 31': 151, 'Junho 1': 152, 'Junho 2': 153, 'Junho 3': 154, 'Junho 4': 155, 'Junho 5': 156, 'Junho 6': 157, 'Junho 7': 158, 'Junho 8': 159, 'Junho 9': 160, 'Junho 10': 161, 'Junho 11': 162, 'Junho 12': 163, 'Junho 13': 164, 'Junho 14': 165, 'Junho 15': 166, 'Junho 16': 167, 'Junho 17': 168, 'Junho 18': 169, 'Junho 19': 170, 'Junho 20': 171, 'Junho 21': 172, 'Junho 22': 173, 'Junho 23': 174, 'Junho 24': 175, 'Junho 25': 176, 'Junho 26': 177, 'Junho 27': 178, 'Junho 28': 179, 'Junho 29': 180, 'Junho 30': 181, 'Julho 1': 182, 'Julho 2': 183, 'Julho 3': 184, 'Julho 4': 185, 'Julho 5': 186, 'Julho 6': 187, 'Julho 7': 188, 'Julho 8': 189, 'Julho 9': 190, 'Julho 10': 191, 'Julho 11': 192, 'Julho 12': 193, 'Julho 13': 194, 'Julho 14': 195, 'Julho 15': 196, 'Julho 16': 197, 'Julho 17': 198, 'Julho 18': 199, 'Julho 19': 200, 'Julho 20': 201, 'Julho 21': 202, 'Julho 22': 203, 'Julho 23': 204, 'Julho 24': 205, 'Julho 25': 206, 'Julho 26': 207, 'Julho 27': 208, 'Julho 28': 209, 'Julho 29': 210, 'Julho 30': 211, 'Julho 31': 212, 'Agosto 1': 213, 'Agosto 2': 214, 'Agosto 3': 215, 'Agosto 4': 216, 'Agosto 5': 217, 'Agosto 6': 218, 'Agosto 7': 219, 'Agosto 8': 220, 'Agosto 9': 221, 'Agosto 10': 222, 'Agosto 11': 223, 'Agosto 12': 224, 'Agosto 13': 225, 'Agosto 14': 226, 'Agosto 15': 227, 'Agosto 16': 228, 'Agosto 17': 229, 'Agosto 18': 230, 'Agosto 19': 231, 'Agosto 20': 232, 'Agosto 21': 233, 'Agosto 22': 234, 'Agosto 23': 235, 'Agosto 24': 236, 'Agosto 25': 237, 'Agosto 26': 238, 'Agosto 27': 239, 'Agosto 28': 240, 'Agosto 29': 241, 'Agosto 30': 242, 'Agosto 31': 243, 'Setembro 1': 244, 'Setembro 2': 245, 'Setembro 3': 246, 'Setembro 4': 247, 'Setembro 5': 248, 'Setembro 6': 249, 'Setembro 7': 250, 'Setembro 8': 251, 'Setembro 9': 252, 'Setembro 10': 253, 'Setembro 11': 254, 'Setembro 12': 255, 'Setembro 13': 256, 'Setembro 14': 257, 'Setembro 15': 258, 'Setembro 16': 259, 'Setembro 17': 260, 'Setembro 18': 261, 'Setembro 19': 262, 'Setembro 20': 263, 'Setembro 21': 264, 'Setembro 22': 265, 'Setembro 23': 266, 'Setembro 24': 267, 'Setembro 25': 268, 'Setembro 26': 269, 'Setembro 27': 270, 'Setembro 28': 271, 'Setembro 29': 272, 'Setembro 30': 273, 'Outubro 1': 274, 'Outubro 2': 275, 'Outubro 3': 276, 'Outubro 4': 277, 'Outubro 5': 278, 'Outubro 6': 279, 'Outubro 7': 280, 'Outubro 8': 281, 'Outubro 9': 282, 'Outubro 10': 283, 'Outubro 11': 284, 'Outubro 12': 285, 'Outubro 13': 286, 'Outubro 14': 287, 'Outubro 15': 288, 'Outubro 16': 289, 'Outubro 17': 290, 'Outubro 18': 291, 'Outubro 19': 292, 'Outubro 20': 293, 'Outubro 21': 294, 'Outubro 22': 295, 'Outubro 23': 296, 'Outubro 24': 297, 'Outubro 25': 298, 'Outubro 26': 299, 'Outubro 27': 300, 'Outubro 28': 301, 'Outubro 29': 302, 'Outubro 30': 303, 'Outubro 31': 304, 'Novembro 1': 305, 'Novembro 2': 306, 'Novembro 3': 307, 'Novembro 4': 308, 'Novembro 5': 309, 'Novembro 6': 310, 'Novembro 7': 311, 'Novembro 8': 312, 'Novembro 9': 313, 'Novembro 10': 314, 'Novembro 11': 315, 'Novembro 12': 316, 'Novembro 13': 317, 'Novembro 14': 318, 'Novembro 15': 319, 'Novembro 16': 320, 'Novembro 17': 321, 'Novembro 18': 322, 'Novembro 19': 323, 'Novembro 20': 324, 'Novembro 21': 325, 'Novembro 22': 326, 'Novembro 23': 327, 'Novembro 24': 328, 'Novembro 25': 329, 'Novembro 26': 330, 'Novembro 27': 331, 'Novembro 28': 332, 'Novembro 29': 333, 'Novembro 30': 334, 'Dezembro 1': 335, 'Dezembro 2': 336, 'Dezembro 3': 337, 'Dezembro 4': 338, 'Dezembro 5': 339, 'Dezembro 6': 340, 'Dezembro 7': 341, 'Dezembro 8': 342, 'Dezembro 9': 343, 'Dezembro 10': 344, 'Dezembro 11': 345, 'Dezembro 12': 346, 'Dezembro 13': 347, 'Dezembro 14': 348, 'Dezembro 15': 349, 'Dezembro 16': 350, 'Dezembro 17': 351, 'Dezembro 18': 352, 'Dezembro 19': 353, 'Dezembro 20': 354, 'Dezembro 21': 355, 'Dezembro 22': 356, 'Dezembro 23': 357, 'Dezembro 24': 358, 'Dezembro 25': 359, 'Dezembro 26': 360, 'Dezembro 27': 361, 'Dezembro 28': 362, 'Dezembro 29': 363, 'Dezembro 30': 364, 'Dezembro 31': 365}
ano_rev = inv_map = {v: k for k, v in ano.items()}
days = 300 - ano[primeiro_infectado]  # days to simulate, starts counting from the first infected
begin = ano[primeiro_infectado]
acoes1 = [ano[i]-begin for i in acoes]  # <ctm> Portuguese: acoes = English: actions <ctm>


# model parameters
parameters = dict(
    seed=999666,  # fixed seed for random operations
    age_dist=age_dist,  # age distribution
    fam_structure=fam_structure,  # family size distribution
    tempo_transporte=tempo_transporte,  # <ctm> time of transport
    qtde_transporte=qtde_transporte,  # <ctm> transport
    qtde_religiao=qtde_religiao,  # <ctm> religion
    layers_0=layers_0,  # model initial layers
    acoes=acoes1,  # <ctm> actions
    layers_tirar=layers_tirar,  # <ctm> layers remove
    layers_por=layers_por,  # <ctm> layers add
    n_nodes=n,  # number of people used to estimate the % of the epidemic
    # the real probabilities are dynamic and depend on several things, which are
    # defined there within the creation of communities. This value here is "how much
    # to consider of these values: 0->nullifies, 1->original, 0.5->half, 2->double
    prob_home=beta_global,  # original> whole family, 3hrs/day. Each layer removed increases home interaction by 25%
    prob_random=beta_global,  # original > 1 next, 1hrs/week, everyone has a chance of having 1 to 10 random connections
    prob_work=beta_global,  # original> 4 next/company size, 6hrs/day, 5 days/week, all 18 to 59 years old.
    prob_transport=beta_global,  # original > 5 next/vehicle size, 1.2hrs/day, 50% population (random)
    prob_school=beta_global,
    # original> 4 next/room size, 5hrs/day, 5 days/week, all population from 0 to 17 years old
    prob_religion=beta_global,  # original> 6 next/church size, 2hrs/week, 40% of the population (random)
    verbose=False  # print or not the information during construction and simulation
)


##################### INFECTION PARAMETERS
infected_t0 = 1  # initial infected
home_isolation = 'home - total'  # 'total' or 'partial', isolate the vertex completely at home (house edges stay), or keep the random ones as well
hospital_isolation = 'hospital - total'  # 'total' or 'partial', isolate the vertex in the hospital entirely or keep the random links
                             #   representing connections with people in the hospital


def fun(data):
    return " ".join([item for var in data for item in var])


def analyze(i):
    parameters['seed'] = i
    np.random.seed(parameters['seed'])
    random.seed(parameters['seed'])
    count=[]
    G = model.createGraph(parameters)
    G, count = simulate(G, parameters, infected_t0=infected_t0, days=days, hospital_isolation=hospital_isolation, home_isolation=home_isolation)
    G = []
    count = np.true_divide(count, n)
    return count


if __name__ == '__main__':
    print("Home network: ", parameters['layers_0'])  # <ctm-code> print("Rede inicial: ", parameters['layers_0'])
    print("Actions days: ", acoes)  # <ctm-code> print("Ações dias: ", acoes)
    print("Layers to insert: ", layers_por)  # <ctm-code> print("Camadas a inserir: ", layers_por)
    print("Layers to be removed: ", layers_tirar)  # <ctm-code> print("Camadas a tirar: ", layers_tirar)
    
    file = 'experimentos/' + cidade + '_REALISTIC_-' + str(parameters['n_nodes']) + '_reps-'
    file = file + str(repetitions) + '_beta-' + str(beta_global) + '_actions-' + fun(acoes)  # <ctm-code> file = file + str(repetitions) + '_beta-' + str(beta_global) + '_acoes-' + fun(acoes)
    file = file + '_add-(' +  fun(layers_por)  # <ctm-code> file = file + '_por-(' +  fun(layers_por)
    file = file + ')_remove-(' + fun(layers_tirar) +  ').pickle'  # <ctm-code> file = file + ')_tirar-(' + fun(layers_tirar) +  ').pickle'
    
    exists = os.path.isfile(file)
    if exists:
        with open(file, 'rb') as f:
            count = pickle.load(f)
            f.close()
    else:
        
        index_list = [i for i in range(1, repetitions+1)]
        processes = os.cpu_count()-2
        pool = Pool(processes)
        print('Running with', processes, 'threads')
        
        count=[]
        t1_start = perf_counter() 
        result = pool.map(analyze, iterable=index_list, chunksize=None)
        # result =analyze(1)

        t1_stop = perf_counter() 
        print("Spent ", t1_stop-t1_start, " seconds")
        
        count = np.zeros((11, days, repetitions))
        # count[:,:,0] = result
        i=0 
        for it in result:    
            count[:,:,i] = it  
            i+=1

        with open(file, 'wb') as f:
            pickle.dump(count, f)
    
    final = np.copy(count) 
    final3 = np.copy(count)*populacao_br
    
    final = final[:,:, final3[0,days-1,:]!=n-1]
    final3 = np.copy(final)*populacao_br
    
    count = np.mean(final, axis=2)  # iteration average
    std_mat = np.std(final, axis=2)  # standard deviation
    
    final2 = np.copy(count)  
    count =count*populacao_br
    std_mat = std_mat*populacao_br

    total_casos = np.cumsum(final[5], axis=0)*populacao_br
    std_und = np.std(total_casos, axis=1)
    und_casos = np.mean(total_casos, axis=1)
    
    total_casos = np.cumsum(final[6], axis=0)*populacao_br
    std_diag = np.std(total_casos, axis=1)
    diag_casos = np.mean(total_casos, axis=1)
    
    total_casos = np.cumsum(final[8], axis=0)*populacao_br
    recovered = np.mean(total_casos, axis=1)
    std_recv =  np.std(total_casos, axis=1)
    
    errorspace = 2
    plt.figure(0)
    plt.rcParams.update({'font.size': 15})
    
    eb=plt.errorbar(range(0, days), und_casos, yerr=std_und, lw=2, color='red', label='não diagnosticado', errorevery=errorspace)
    eb[-1][0].set_linewidth(1)
    
    eb=plt.errorbar(range(0, days), diag_casos, yerr=std_diag, lw=2, color='orange', label='diagnosticado', errorevery=errorspace)
    eb[-1][0].set_linewidth(1)
    
    eb=plt.errorbar(range(0, days), recovered, yerr=std_recv, lw=2, alpha=0.65, color='green', label='recuperado', errorevery=errorspace)
    eb[-1][0].set_linewidth(1)
    
    # plt.ylim(0, 5100000)
    # plt.xlim([0,210])
    plt.xlabel('Days since first case')  # <ctm-code> plt.xlabel('Dias desde o primeiro caso')
    plt.ylabel('Total cases')  # <ctm-code> plt.ylabel('Total de casos')
    ax = plt.gca()
    # ax.yaxis.set_major_formatter(ticker.EngFormatter())

    plt.xticks([0, acoes1[0], days],  ("0\n"+primeiro_infectado, str(acoes1[0]) + "\n" + acoes[0], str(days) +"\nDezembro 31"))

    # plt.yticks([])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig(path2 + "remove2_casos_total.pdf", dpi=500)
    
    
    
    plt.figure(1)
    plt.rcParams.update({'font.size': 15})
    
    # plt.plot((count[9]), label='leito', color='orange')
    # plt.plot((count[10]), label='UTI', color='red')    
    # plt.xticks([0, 34, 64, 95, 125, 156, 186, 217, 247, 278],  ('1º', 'Abr.', 'Maio', 'Jun.', "Jul.", "Ag.", "Set.", "Out.", "Nov.", "Dez."))

    peak = ((np.where(count[9] == np.amax(count[9])))[0] + (np.where(count[10] == np.amax(count[10])))[0])/2
    peak = int(peak[0])
    ICUbeds = np.round(count[10, peak])
    bedsstd = np.round(count[9, peak])
    print("Peak of occupied beds: ", ano_rev[ano[primeiro_infectado] + peak]  ," - UTIs ", ICUbeds, "(+-", np.round(std_mat[10,peak]) ,") - normal ", bedsstd, "(+-", np.round(std_mat[9,peak]) ,")")  # <ctm-code> print("Pico de leitos ocupados: ", ano_rev[ano[primeiro_infectado] + peak]  ," - UTIs ", ICUbeds, "(+-", np.round(std_mat[10,peak]) ,") - normal ", bedsstd, "(+-", np.round(std_mat[9,peak]) ,")")
    # plt.ylim(0, 280000)
    # plt.yticks([0, 43000, 86000, 135000, 215000])
    
    plt.errorbar(range(0, days), (count[9]), yerr=std_mat[9], label='normal', color='orange',zorder=0,errorevery=errorspace)
    plt.errorbar(range(0, days), (count[10]), yerr=std_mat[10], label='UTI/Respirator', color='red',zorder=0,errorevery=errorspace)  # <ctm-code> plt.errorbar(range(0, days), (count[10]), yerr=std_mat[10], label='UTI/Respirador', color='red',zorder=0,errorevery=errorspace)
    plt.grid(zorder=10)
    plt.xlabel('Days since first case')  # <ctm-code> plt.xlabel('Dias desde o primeiro caso')
    plt.ylabel('Occupied beds')  # <ctm-code> plt.ylabel('Leitos ocupados')
    
    plt.xticks([0, peak, days],  ("0\n"+primeiro_infectado, str(peak) + "\n" + ano_rev[begin+peak], str(days) +"\nDezembro 31"))

    # plt.xlabel('Vagas ocupadas')
    # plt.ylabel('Total')
    # plt.xlim([0,210])
    plt.legend(loc='upper left')
    ax = plt.gca()
    # ax.yaxis.set_major_formatter(ticker.EngFormatter())
    plt.tight_layout()
    plt.show()
    # plt.savefig(path2 + "donothing_leitos.pdf", dpi=500)
    
    
    data = count[6]
    plt.figure(2)
    plt.rcParams.update({'font.size': 16})
    # fig, (ax1) = plt.subplots(2)
    
    # ax1.scatter(250,1, label="no new cases")
    plt.bar(range(1, len(data)+1), data, color='blue',width=1.0)

    # plt.ylim(0, 41000)
    # plt.yticks([0,8700, 20000, 30000, 40000])

    peak = (np.where(data == np.amax(data)))[0]
    plt.xticks([0, peak[0], days],  ("0\n"+primeiro_infectado, str(peak[0]) + "\n" + ano_rev[begin+peak[0]], str(days) +"\nDezembro 31"))

    plt.xlabel('Days since first case')  # <ctm-code> plt.xlabel('Dias desde o primeiro caso')
    plt.ylabel('New cases')  # <ctm-code> plt.ylabel('Novos casos')
    plt.grid()
    # plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig(path2 + "casos_daily.pdf", dpi=500)

    
    data = count[7]
    plt.figure(3)
    plt.rcParams.update({'font.size': 16})
    # fig, (ax1) = plt.subplots(3)
    
    # ax1.scatter(281,0, label="no new deaths")
    plt.bar(range(1, len(data)+1), data, color='red', width=1.0)
    
    # plt.ylim(0, 8000)
    peak = (np.where(data == np.amax(data)))[0]
    
    plt.xticks([0, peak[0], days],  ("0\n"+primeiro_infectado, str(peak[0]) + "\n" + ano_rev[begin+peak[0]], str(days) +"\nDezembro 31"))

    plt.xlabel('Days since first case')  # <ctm-code> plt.xlabel('Dias desde o primeiro caso')
    plt.ylabel('New cases')  # <ctm-code> plt.ylabel('Novos casos')
    plt.grid()
    plt.tight_layout()
    plt.show()
    # plt.savefig(path2 + "mortes_daily.pdf", dpi=500)


   
    casos = np.sum(final[6], axis=0)*populacao_br
    casos = casos[casos != 0]
    casos2 = np.sum(final[5], axis=0)*populacao_br
    casos2 = casos2[casos2 != 0]
    mortes = np.sum(final[7], axis=0)*populacao_br
    mortes = mortes[mortes != 0]
    
    print('Statistics according to the total population:')  # <ctm-code> print('Estatistics de acordo com a população total:')
    print('Undiagnosed cases:        ',  np.round(np.mean(casos2)), '(+-', np.round(np.std(casos2)),')' ,' -  (', np.round(max(np.cumsum(final2[5]))*100, decimals=4), '%)')  # <ctm-code> print('Casos não diagnosticados: ',  np.round(np.mean(casos2)), '(+-', np.round(np.std(casos2)),')' ,' -  (', np.round(max(np.cumsum(final2[5]))*100, decimals=4), '%)')
    print('Diagnosed cases:          ',  np.round(np.mean(casos)), '(+-', np.round(np.std(casos)),')' ,' -  (', np.round(max(np.cumsum(final2[6]))*100, decimals=4), '%)')  # <ctm-code> print('Casos diagnosticados:     ',  np.round(np.mean(casos)), '(+-', np.round(np.std(casos)),')' ,' -  (', np.round(max(np.cumsum(final2[6]))*100, decimals=4), '%)')
    print('Deaths:                   ',  np.round(np.mean(mortes)), '(+-', np.round(np.std(mortes)),')' ,' -  (', np.round(max(np.cumsum(final2[7]))*100, decimals=4), '%)')  # <ctm-code> print('Mortes:                   ',  np.round(np.mean(mortes)), '(+-', np.round(np.std(mortes)),')' ,' -  (', np.round(max(np.cumsum(final2[7]))*100, decimals=4), '%)')
    print('Recovered:                ',  (max(np.cumsum(count[8]))), ' (', np.round(max(np.cumsum(final2[8]))*100, decimals=4), '%)')  # <ctm-code> print('Recuperados:              ',  (max(np.cumsum(count[8]))), ' (', np.round(max(np.cumsum(final2[8]))*100, decimals=4), '%)')
    
    
    
    
    
    
    
    
    
    
    
   
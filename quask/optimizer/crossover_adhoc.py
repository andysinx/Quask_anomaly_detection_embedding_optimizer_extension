import numpy as np
import random

def heterogenous_crossover(parents, offspring_size, ga_instance):
   
            
    def split_and_convert_to_tuples(lst, chunk_size):
        # Dividi la lista in sotto-liste di dimensione chunk_size
        chunks = [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
        
        # Converti le sotto-liste in tuple
        tuples = [tuple(sublist) for sublist in chunks]
        
        return tuples
    
    def uniform_crossover(parent1, parent2, crossover_rate=0.5):
        # Initialize offspring
        
        child = []
        for i in range(len(parent1)):
            if np.random.rand() < crossover_rate:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child
    
    def heterogeneous_crossover(parent1, parent2):
        print("parent_1_format: ", parent1)
        print("parent_2_format: ", parent2)
        
        offsprings = []
        
        circuit_parent_1 = {
            'generators': [],
            'qbits_1': [],
            'qbits_2': [],
            'features': [],
            'bandwidths': []
        }
    
        circuit_parent_2 = {
            'generators': [],
            'qbits_1': [],
            'qbits_2': [],
            'features': [],
            'bandwidths': []
        }

        for tup in parent1:
                circuit_parent_1['generators'].append(tup[0])
                circuit_parent_1['qbits_1'].append(tup[1])
                circuit_parent_1['qbits_2'].append(tup[2])
                circuit_parent_1['features'].append(tup[3])
                circuit_parent_1['bandwidths'].append(tup[4])
        
        for tup in parent2:
            circuit_parent_2['generators'].append(tup[0])
            circuit_parent_2['qbits_1'].append(tup[1])
            circuit_parent_2['qbits_2'].append(tup[2])
            circuit_parent_2['features'].append(tup[3])
            circuit_parent_2['bandwidths'].append(tup[4])
            
        print("circuit_parent_1: ", circuit_parent_1)
        print("circuit_parent_2: ", circuit_parent_2)
        
        
        for i in range(len(ga_instance.initial_population) - 1):
            offsprings.append({
                'generators': uniform_crossover(circuit_parent_1['generators'],circuit_parent_2['generators'], 0.5),
                'qbits_1': uniform_crossover(circuit_parent_1['qbits_1'],circuit_parent_2['qbits_1'], 0.5),
                'qbits_2': uniform_crossover(circuit_parent_1['qbits_2'],circuit_parent_2['qbits_2'], 0.5),
                'features': uniform_crossover(circuit_parent_1['features'],circuit_parent_2['features'], 0.5),
                'bandwidths': circuit_parent_1['bandwidths'] 
            })
        
        print("offsprings_modified:", offsprings)
        offsprings_final = []
        
        for i in range(len(offsprings)):
            temp = []
            for j in range(len(offsprings[i]['generators'])):
                print("index: ",j)
                temp.append(offsprings[i]['generators'][j])    
                temp.append(offsprings[i]['qbits_1'][j])
                temp.append(offsprings[i]['qbits_2'][j])    
                temp.append(offsprings[i]['features'][j])    
                temp.append(offsprings[i]['bandwidths'][j])
                print("TEMP: ",temp)           
            offsprings_final.append(temp)
    
        print("offsprings_final: ", offsprings_final)
        
        return offsprings_final
            
    
    offsprings = heterogeneous_crossover(parent1=split_and_convert_to_tuples(parents[0],5),parent2=split_and_convert_to_tuples(parents[1],5)) 
    return np.array(offsprings)





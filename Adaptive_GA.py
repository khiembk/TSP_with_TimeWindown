import random
from copy import deepcopy
import time
def read_input_from_file(filename):
    try:
        with open(filename, 'r') as file:
            # Read n
            line = file.readline().strip()
            n = int(line)
            if not 1 <= n <= 1000:
                raise ValueError("N must be between 1 and 1000")
            
            time_arrivals = []
            times = []
            max_time = 0
            
            # Read time windows and service times
            for _ in range(n):
                line = file.readline().strip()
                if not line:
                    raise ValueError("Unexpected end of file while reading time windows")
                e, l, d = map(int, line.split())
                if e < 0 or l < e or d < 0:
                    raise ValueError("Invalid time window or service time")
                time_arrivals.append([e, l, d])
                max_time = max(max_time, l)
            
            # Read travel time matrix
            for _ in range(n+1):
                line = file.readline().strip()
                if not line:
                    raise ValueError("Unexpected end of file while reading travel time matrix")
                row = list(map(int, line.split()))
                if len(row) != n+1 or any(t < 0 for t in row):
                    raise ValueError("Invalid travel time matrix row")
                times.append(row)
            
            # Check for extra data
            if file.readline().strip():
                raise ValueError("Extra data found after expected input")
            
            return n, time_arrivals, times, max_time
    except FileNotFoundError:
        raise ValueError(f"Input file not found: {filename}")
    except ValueError as e:
        raise ValueError(f"Input error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error reading file: {str(e)}")
def read_input():
    try:
        n = int(input())
        if not 1 <= n <= 1000:
            raise ValueError("N must be between 1 and 1000")
        
        time_arrivals = []
        times = []
        max_time = 0
        
        for _ in range(n):
            e, l, d = map(int, input().split())
            if e < 0 or l < e or d < 0:
                raise ValueError("Invalid time window or service time")
            time_arrivals.append([e, l, d])
            max_time = max(max_time, l)
        
        for _ in range(n+1):
            row = list(map(int, input().split()))
            if len(row) != n+1 or any(t < 0 for t in row):
                raise ValueError("Invalid travel time matrix row")
            times.append(row)
        
        return n, time_arrivals, times, max_time
    except ValueError as e:
        raise ValueError(f"Input error: {str(e)}")

def compute_cost(n, time_arrivals, times, chromosome):
    if len(chromosome) != n or not all(1 <= x <= n for x in chromosome) or len(set(chromosome)) != n:
        return float('inf')
    
    cost = 0
    current = 0
    for i in range(n):
        cost += times[current][chromosome[i]]
        current = chromosome[i]
    
    cost += times[current][0]
    return cost

def compute_penalty(n, time_arrivals, times, chromosome):
    if len(chromosome) != n or not all(1 <= x <= n for x in chromosome) or len(set(chromosome)) != n:
        return float('inf')
    
    time_arrive = []
    cur_time = 0
    cur_node = 0
    for i in range(n):
        customer = chromosome[i]
        cur_time += times[cur_node][customer]
        if cur_time < time_arrivals[customer-1][0]:
            cur_time = time_arrivals[customer-1][0]
        time_arrive.append(cur_time)
        cur_time += time_arrivals[customer-1][2]
        cur_node = customer
    
    penalty = 0
    for i in range(n):
        customer = chromosome[i]
        if time_arrive[i] > time_arrivals[customer-1][1]:
            penalty += time_arrive[i] - time_arrivals[customer-1][1]
    
    return penalty

def compute_fitness(n, time_arrivals, times, chromosome, cur_ep, total_ep, epsilon=0.01, big_M=100000):
    cost = compute_cost(n, time_arrivals, times, chromosome)
    penalty = compute_penalty(n, time_arrivals, times, chromosome)
    return cost + big_M * (cur_ep / total_ep + epsilon) * penalty

def generate_population(n, pop_size):
    population = []
    for _ in range(pop_size):
        chromosome = generate_random_chromosome(n)
        population.append(chromosome)
    return population

def generate_random_chromosome(n):
    chromosome = list(range(1, n+1))
    random.shuffle(chromosome)
    if len(chromosome) != n or len(set(chromosome)) != n or not all(1 <= x <= n for x in chromosome):
        raise ValueError(f"Invalid chromosome generated: {len(chromosome)}")
    return chromosome

def tournament_selection(population, fitnesses, tournament_size=15):
    if not population or len(population) < tournament_size:
        raise ValueError("Population too small for tournament selection")
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    return min(selected, key=lambda x: x[1])[0]

def order_crossover(parent1, parent2):
    n = len(parent1)
    if len(parent2) != n or len(set(parent1)) != n or len(set(parent2)) != n:
        raise ValueError(f"Invalid parents: parent1={len(parent1)}, parent2={len(parent2)}")
    
    start, end = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[start:end+1] = parent1[start:end+1]
    
    # Filter remaining genes from parent2
    remaining = [x for x in parent2 if x not in child[start:end+1]]
    if len(remaining) != n - (end - start + 1):
        raise ValueError(f"Invalid remaining genes: {remaining}, expected length {n - (end - start + 1)}")
    
    idx = 0
    for i in range(n):
        if child[i] == -1:
            if idx >= len(remaining):
                raise ValueError(f"Index out of range: idx={idx}, remaining={remaining}")
            child[i] = remaining[idx]
            idx += 1
    
    if len(set(child)) != n or any(x < 1 or x > n for x in child):
        raise ValueError(f"Invalid child generated: {len(child)}")
    return child

def pmx_crossover(parent1, parent2):
    n = len(parent1)
    # Validate parents
    if len(parent2) != n or len(set(parent1)) != n or len(set(parent2)) != n:
        raise ValueError(f"Invalid parents: parent1={len(parent1)}, parent2={len(parent2)}")
    
    # Select crossover points and initialize child
    start, end = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[start:end+1] = parent1[start:end+1]
    
    # Track used genes
    used = set(child[start:end+1])
    
    # Create mapping from parent1 to parent2 in the segment
    mapping = {parent1[i]: parent2[i] for i in range(start, end+1)}
    
    # Fill positions outside the segment
    for i in range(n):
        if i < start or i > end:
            gene = parent2[i]
            # Resolve conflicts using mapping
            visited = set()
            while gene in used and gene not in visited:
                visited.add(gene)
                gene = mapping.get(gene, gene)
            # If gene is still used, pick an unused gene
            if gene in used:
                available = set(range(1, n+1)) - used
                gene = available.pop()
            child[i] = gene
            used.add(gene)
    
    # Validate child
    if len(set(child)) != n or any(x < 1 or x > n for x in child):
        raise ValueError(f"Invalid child generated: {len(child)}")
    return child

def cycle_crossover(parent1, parent2):
    n = len(parent1)
    if len(parent2) != n or len(set(parent1)) != n or len(set(parent2)) != n:
        raise ValueError(f"Invalid parents: parent1={len(parent1)}, parent2={len(parent2)}")
    
    child = [-1] * n
    visited = [False] * n
    pos = 0
    cycle = []
    while not visited[pos]:
        cycle.append(pos)
        visited[pos] = True
        gene = parent1[pos]
        pos = parent2.index(gene)
    
    for i in cycle:
        child[i] = parent1[i]
    
    for i in range(n):
        if child[i] == -1:
            child[i] = parent2[i]
    
    if len(set(child)) != n or any(x < 1 or x > n for x in child):
        raise ValueError(f"Invalid child generated: {child}")
    return child

def cycle_crossover_two_child(parent1, parent2):
    n = len(parent1)
    if len(parent2) != n or len(set(parent1)) != n or len(set(parent2)) != n:
        raise ValueError(f"Invalid parents: parent1={len(parent1)}, parent2={len(parent2)}")
    
    # Child 1: Cycles from parent1, fill with parent2
    child1 = [-1] * n
    visited = [False] * n
    pos = 0
    cycle = []
    while not visited[pos]:
        cycle.append(pos)
        visited[pos] = True
        gene = parent1[pos]
        pos = parent2.index(gene)
    for i in cycle:
        child1[i] = parent1[i]
    for i in range(n):
        if child1[i] == -1:
            child1[i] = parent2[i]
    
    # Child 2: Cycles from parent2, fill with parent1
    child2 = [-1] * n
    visited = [False] * n
    pos = 0
    cycle = []
    while not visited[pos]:
        cycle.append(pos)
        visited[pos] = True
        gene = parent2[pos]
        pos = parent1.index(gene)
    for i in cycle:
        child2[i] = parent2[i]
    for i in range(n):
        if child2[i] == -1:
            child2[i] = parent1[i]
    
    # Validate both children
    if len(set(child1)) != n or any(x < 1 or x > n for x in child1):
        raise ValueError(f"Invalid child1 generated: {len(child1)}")
    if len(set(child2)) != n or any(x < 1 or x > n for x in child2):
        raise ValueError(f"Invalid child2 generated: {len(child2)}")
    
    return child1, child2


def pmx_crossover_two_child(parent1, parent2):
    n = len(parent1)
    if len(parent2) != n or len(set(parent1)) != n or len(set(parent2)) != n:
        raise ValueError(f"Invalid parents: parent1={len(parent1)}, parent2={len(parent2)}")
    
    start, end = sorted(random.sample(range(n), 2))
    
    # Child 1: Segment from parent1, fill with parent2
    child1 = [-1] * n
    child1[start:end+1] = parent1[start:end+1]
    used1 = set(child1[start:end+1])
    mapping1 = {parent1[i]: parent2[i] for i in range(start, end+1)}
    for i in range(n):
        if i < start or i > end:
            gene = parent2[i]
            visited = set()
            while gene in used1 and gene not in visited:
                visited.add(gene)
                gene = mapping1.get(gene, gene)
            if gene in used1:
                available = set(range(1, n+1)) - used1
                gene = available.pop()
            child1[i] = gene
            used1.add(gene)
    
    # Child 2: Segment from parent2, fill with parent1
    child2 = [-1] * n
    child2[start:end+1] = parent2[start:end+1]
    used2 = set(child2[start:end+1])
    mapping2 = {parent2[i]: parent1[i] for i in range(start, end+1)}
    for i in range(n):
        if i < start or i > end:
            gene = parent1[i]
            visited = set()
            while gene in used2 and gene not in visited:
                visited.add(gene)
                gene = mapping2.get(gene, gene)
            if gene in used2:
                available = set(range(1, n+1)) - used2
                gene = available.pop()
            child2[i] = gene
            used2.add(gene)
    
    # Validate both children
    if len(set(child1)) != n or any(x < 1 or x > n for x in child1):
        raise ValueError(f"Invalid child1 generated: {len(child1)}")
    if len(set(child2)) != n or any(x < 1 or x > n for x in child2):
        raise ValueError(f"Invalid child2 generated: {len(child2)}")
    
    return child1, child2

def order_crossover_two_child(parent1, parent2):
    n = len(parent1)
    if len(parent2) != n or len(set(parent1)) != n or len(set(parent2)) != n:
        raise ValueError(f"Invalid parents: parent1={len(parent1)}, parent2={len(parent2)}")
    
    start, end = sorted(random.sample(range(n), 2))
    
    # Child 1: Segment from parent1, fill with parent2
    child1 = [-1] * n
    child1[start:end+1] = parent1[start:end+1]
    remaining1 = [x for x in parent2 if x not in child1[start:end+1]]
    if len(remaining1) != n - (end - start + 1):
        raise ValueError(f"Invalid remaining genes for child1: {remaining1}")
    idx = 0
    for i in range(n):
        if child1[i] == -1:
            if idx >= len(remaining1):
                raise ValueError(f"Index out of range for child1: idx={idx}, remaining={remaining1}")
            child1[i] = remaining1[idx]
            idx += 1
    
    # Child 2: Segment from parent2, fill with parent1
    child2 = [-1] * n
    child2[start:end+1] = parent2[start:end+1]
    remaining2 = [x for x in parent1 if x not in child2[start:end+1]]
    if len(remaining2) != n - (end - start + 1):
        raise ValueError(f"Invalid remaining genes for child2: {remaining2}")
    idx = 0
    for i in range(n):
        if child2[i] == -1:
            if idx >= len(remaining2):
                raise ValueError(f"Index out of range for child2: idx={idx}, remaining={remaining2}")
            child2[i] = remaining2[idx]
            idx += 1
    
    # Validate both children
    if len(set(child1)) != n or any(x < 1 or x > n for x in child1):
        raise ValueError(f"Invalid child1 generated: {len(child1)}")
    if len(set(child2)) != n or any(x < 1 or x > n for x in child2):
        raise ValueError(f"Invalid child2 generated: {len(child2)}")
    
    return child1, child2



def mutate(chromosome, n, time_arrivals, times, cur_ep, total_ep, delta=20, mutation_rate=0.01, max_time=10):
    if random.random() >= mutation_rate:
        return chromosome
    
    old_penalty = compute_penalty(n, time_arrivals, times, chromosome)
    original_chromosome = chromosome.copy()
    attempts = 0
    
    while attempts < max_time:
        new_chromosome = chromosome.copy()
        i, j = random.sample(range(len(new_chromosome)), 2)
        new_chromosome[i], new_chromosome[j] = new_chromosome[j], new_chromosome[i]
        
        new_penalty = compute_penalty(n, time_arrivals, times, new_chromosome)
        if new_penalty <= old_penalty + delta:
            return new_chromosome
        attempts += 1
    
    return original_chromosome

def select_population(n, time_arrivals, times, cur_ep, total_ep, population, new_population, pop_size, reserve_pop_size):
    combined = [(deepcopy(chrom), compute_fitness(n, time_arrivals, times, chrom, cur_ep, total_ep)) 
                for chrom in population + new_population]
    
    combined.sort(key=lambda x: x[1])
    new_population = [chrom for chrom, _ in combined[:pop_size]]
    reserve_pop = combined[pop_size:pop_size + reserve_pop_size]
    
    return new_population, reserve_pop

def random_index_method(crossover_score):
    total = sum(crossover_score)
    p_order = crossover_score[0]/total
    p_cycle = crossover_score[1]/total

    p = random.random()
    if p<= p_order:
        return 0
    if p_order <p and p<= p_order + p_cycle:
        return 1
    else:
        return 2

def select_random_instances(population):
    seleted = random.sample(population,k=1)
    instance = seleted[0]
    if len(instance) ==2:
        return instance[0]
    else:
        return instance


def get_top_k_elite(population, fitnesses, k):
    # Input validation
    if not population or len(population) < k:
        raise ValueError(f"Population size ({len(population)}) must be at least {k}")
    if k <= 0:
        raise ValueError("k must be positive")
    
    # Pair population with fitnesses and sort by fitness (ascending, lower is better)
    population_fitness = list(zip(population, fitnesses))
    population_fitness.sort(key=lambda x: x[1])
    
    # Return the top k individuals (without fitness values)
    return [individual for individual, _ in population_fitness[:k]]
    
def genetic_algorithm(n, time_arrivals, times, max_time, pop_size=170, reserve_pop_size=30, total_ep=1500, mutation_rate=0.01, p_reserve=0.1):
    population = generate_population(n, pop_size)
    reserve_pop = generate_population(n, reserve_pop_size)
    crossover_methods = [order_crossover_two_child, cycle_crossover_two_child, pmx_crossover_two_child]
    best_solution = None
    best_fitness = float('inf')

    crossover_score = [1]*3
    method_indx = 0

    for ep in range(total_ep):
        fitnesses = [compute_fitness(n, time_arrivals, times, chrom, ep, total_ep) for chrom in population]
        
        min_fitness = min(fitnesses)
        if best_fitness != float('inf'):
            best_fitness = compute_fitness(n, time_arrivals, times, best_solution, ep, total_ep)

        if min_fitness < best_fitness:
            if best_fitness!= float('inf'):
                add_score = (best_fitness - min_fitness)/best_fitness
                crossover_score[method_indx] += add_score

            best_fitness = min_fitness
            best_solution = population[fitnesses.index(min_fitness)].copy()
            
        method_indx = random_index_method(crossover_score)
        crossover = crossover_methods[method_indx]
        new_population = []
        for _ in range(pop_size):
            try:
                
                parent1 = tournament_selection(population, fitnesses)
                
                if random.random() < p_reserve*(1 - ep/total_ep) and reserve_pop:
                    parent2 = select_random_instances(reserve_pop)
                else:
                    parent2 = tournament_selection(population, fitnesses)
                
                #crossover = random.choice(crossover_methods)
                
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, n, time_arrivals, times, ep, total_ep, mutation_rate=mutation_rate)
                child2 = mutate(child2,n, time_arrivals, times, ep, total_ep, mutation_rate=mutation_rate)
                new_population.append(child1)
                new_population.append(child2)
            except Exception as e:
                print(f"Error in crossover/mutation: {str(e)}")
                continue
        
        population, reserve_pop = select_population(n, time_arrivals, times, ep, total_ep, population, new_population, pop_size, reserve_pop_size)
    
    print("score each crossover methods: ", crossover_score)
    return best_solution

if __name__ == "__main__":
    try:
        n, time_arrivals, times, max_time = read_input_from_file("tests/test8/input.in")
        start_time = time.time()
        best_route = genetic_algorithm(n, time_arrivals, times, max_time)
        end_time = time.time()
        print(n)
        cost = compute_cost(n, time_arrivals, times, best_route)
        print("solution value", cost if cost != float('inf') else "Invalid solution")
        print(*best_route)
        print(f"Execution time: {end_time - start_time:.4f} seconds")
    except ValueError as e:
        print(f"Error: {str(e)}")
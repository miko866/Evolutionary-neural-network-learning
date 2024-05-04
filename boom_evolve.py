# -*- coding: utf-8 -*-
import pygame
import numpy as np
import random
import math
from deap import base
from deap import creator
from deap import tools

pygame.font.init()

#-----------------------------------------------------------------------------
# Parametry hry 
#-----------------------------------------------------------------------------

WIDTH, HEIGHT = 900, 500
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
WHITE = (255, 255, 255)
TITLE = "Boom Master"
pygame.display.set_caption(TITLE)
FPS = 80
ME_VELOCITY = 5
MAX_MINE_VELOCITY = 3
BOOM_FONT = pygame.font.SysFont("comicsans", 100)
LEVEL_FONT = pygame.font.SysFont("comicsans", 20)
ENEMY_IMAGE  = pygame.image.load("mine.png")
ME_IMAGE = pygame.image.load("me.png")
SEA_IMAGE = pygame.image.load("sea.png")
FLAG_IMAGE = pygame.image.load("flag.png")
ENEMY_SIZE = 50
ME_SIZE = 50
ENEMY = pygame.transform.scale(ENEMY_IMAGE, (ENEMY_SIZE, ENEMY_SIZE))
ME = pygame.transform.scale(ME_IMAGE, (ME_SIZE, ME_SIZE))
SEA = pygame.transform.scale(SEA_IMAGE, (WIDTH, HEIGHT))
FLAG = pygame.transform.scale(FLAG_IMAGE, (ME_SIZE, ME_SIZE))

# ----------------------------------------------------------------------------
# třídy objektů 
# ----------------------------------------------------------------------------

# trida reprezentujici minu
class Mine:
    def __init__(self):

        # random x direction
        if random.random() > 0.5:
            self.dirx = 1
        else:
            self.dirx = -1

        # random y direction    
        if random.random() > 0.5:
            self.diry = 1
        else:
            self.diry = -1

        x = random.randint(200, WIDTH - ENEMY_SIZE)
        y = random.randint(200, HEIGHT - ENEMY_SIZE)
        self.rect = pygame.Rect(x, y, ENEMY_SIZE, ENEMY_SIZE)

        self.velocity = random.randint(1, MAX_MINE_VELOCITY)


# trida reprezentujici me, tedy meho agenta        
class Me:
    def __init__(self):
        self.rect = pygame.Rect(10, random.randint(1, 300), ME_SIZE, ME_SIZE)
        self.alive = True
        self.won = False
        self.timealive = 0
        self.sequence = []
        self.fitness = 0
        self.dist = 0
        self.last_position = (self.rect.x, self.rect.y)
        self.is_moving = True
        self.time_still = 0
        self.velocity = 0

# třída reprezentující cíl = praporek    
class Flag:
    def __init__(self):
        self.rect = pygame.Rect(WIDTH - ME_SIZE, HEIGHT - ME_SIZE - 10, ME_SIZE, ME_SIZE)


# třída reprezentující nejlepšího jedince - hall of fame   
class Hof:
    def __init__(self):
        self.sequence = []

# -----------------------------------------------------------------------------    
# nastavení herního plánu    
#-----------------------------------------------------------------------------
# rozestavi miny v danem poctu num
def set_mines(num):
    l = []
    for i in range(num):
        m = Mine()
        l.append(m)
    return l

# inicializuje me v poctu num na start 
def set_mes(num):
    l = []
    for i in range(num):
        m = Me()
        l.append(m)

    return l

# zresetuje vsechny mes zpatky na start
def reset_mes(mes, pop):
    for me, sequence in zip(mes, pop):
        me.rect.x = 10
        me.rect.y = 10
        me.alive = True
        me.dist = 0
        me.won = False
        me.timealive = 0
        me.sequence = sequence
        me.fitness = 0

# -----------------------------------------------------------------------------    
# senzorické funkce 
# -----------------------------------------------------------------------------    

# Define direction constants
RIGHT = 0
LEFT = 180
UP = 90
DOWN = 270
MAX_DISTANCE = math.sqrt(WIDTH**2 + HEIGHT**2)

# Calculate distance and direction from one rectangle to another
# Return distance and direction in degrees
# Direction is None if the rectangles are at the same position
def calculate_distance_and_direction(from_rect, to_rect):
    dx = to_rect.x - from_rect.x
    dy = to_rect.y - from_rect.y
    distance = (dx**2 + dy**2) / MAX_DISTANCE**2
    direction = None if dx == 0 and dy == 0 else RIGHT if dx > 0 else LEFT if abs(dx) > abs(dy) else UP if dy > 0 else DOWN
    return distance, direction

# Update the agent's position
# Save the last position and check if the agent is moving
# If the agent is not moving, increase the time still counter and do not update the last position
# If the agent is moving, reset the time still counter
# If the agent is moving, update the last position
def senzor(me, mines, flag):
    distances_and_directions = [calculate_distance_and_direction(me.rect, mine.rect) for mine in mines]
    nearest_mine_distance, nearest_mine_dir = min(distances_and_directions, key=lambda x: x[0])
    second_nearest_mine_distance, second_nearest_mine_dir = min((d, dir) for d, dir in distances_and_directions if d > nearest_mine_distance) if len(distances_and_directions) > 1 else (None, None)
    flag_distance, flag_dir = calculate_distance_and_direction(me.rect, flag.rect)
    update(me)
    dist_to_left_wall = me.rect.x / WIDTH
    dist_to_right_wall = (WIDTH - me.rect.x) / WIDTH
    dist_to_top_wall = me.rect.y / HEIGHT
    dist_to_bottom_wall = (HEIGHT - me.rect.y) / HEIGHT
    return [nearest_mine_distance, nearest_mine_dir, second_nearest_mine_distance, second_nearest_mine_dir, flag_distance, flag_dir, dist_to_left_wall, dist_to_right_wall, dist_to_top_wall, dist_to_bottom_wall, me.velocity]

# ---------------------------------------------------------------------------
# funkce řešící pohyb agentů
# ----------------------------------------------------------------------------

# konstoluje kolizi 1 agenta s minama, pokud je kolize vraci True
def me_collision(me, mines):
    for mine in mines:
        if me.rect.colliderect(mine.rect):
            #pygame.event.post(pygame.event.Event(ME_HIT))
            return True
    return False

# kolidujici agenti jsou zabiti, a jiz se nebudou vykreslovat
def mes_collision(mes, mines):
    for me in mes:
        if me.alive and not me.won:
            if me_collision(me, mines):
                me.alive = False


# vraci True, pokud jsou vsichni mrtvi Dave            
def all_dead(mes):
    for me in mes:
        if me.alive:
            return False

    return True


# vrací True, pokud již nikdo nehraje - mes jsou mrtví nebo v cíli
def nobodys_playing(mes):
    for me in mes:
        if me.alive and not me.won:
            return False

    return True


# rika, zda agent dosel do cile
def me_won(me, flag):
    if me.rect.colliderect(flag.rect):
        return True

    return False


# vrací počet živých mes
def alive_mes_num(mes):
    c = 0
    for me in mes:
        if me.alive:
            c += 1
    return c



# vrací počet mes co vyhráli
def won_mes_num(mes):
    c = 0
    for me in mes:
        if me.won:
            c += 1
    return c

 
# resi pohyb miny        
def handle_mine_movement(mine):
    new_x = mine.rect.x + mine.dirx * mine.velocity
    new_y = mine.rect.y + mine.diry * mine.velocity
    mine.dirx = -1 if new_x + mine.rect.width + mine.velocity > WIDTH else 1 if new_x - mine.velocity < 0 else mine.dirx
    mine.diry = -1 if new_y + mine.rect.height + mine.velocity > HEIGHT else 1 if new_y - mine.velocity < 0 else mine.diry
    mine.rect.x = new_x
    mine.rect.y = new_y


# resi pohyb min
def handle_mines_movement(mines):
    for mine in mines:
        handle_mine_movement(mine)


#----------------------------------------------------------------------------
# vykreslovací funkce 
#----------------------------------------------------------------------------


# vykresleni okna
def draw_window(mes, mines, flag, level, generation, timer):
    WIN.blit(SEA, (0, 0))
    for text, pos in [("level: " + str(level), (10, HEIGHT - 30)), ("generation: " + str(generation), (150, HEIGHT - 30)), ("alive: " + str(alive_mes_num(mes)), (350, HEIGHT - 30)), ("won: " + str(won_mes_num(mes)), (500, HEIGHT - 30)), ("timer: " + str(timer), (650, HEIGHT - 30))]:
        t = LEVEL_FONT.render(text, 1, WHITE)
        WIN.blit(t, pos)
    WIN.blit(FLAG, (flag.rect.x, flag.rect.y))
    for mine in mines:
        WIN.blit(ENEMY, (mine.rect.x, mine.rect.y))
    for me in mes:
        if me.alive:
            WIN.blit(ME, (me.rect.x, me.rect.y))
    pygame.display.update()



def draw_text(text):
    t = BOOM_FONT.render(text, 1, WHITE)
    WIN.blit(t, (WIDTH // 2  , HEIGHT // 2))

    pygame.display.update()
    pygame.time.delay(1000)


#-----------------------------------------------------------------------------
# funkce reprezentující neuronovou síť, pro inp vstup a zadané váhy wei, vydá
# čtveřici výstupů pro nahoru, dolu, doleva, doprava    
#----------------------------------------------------------------------------


# <----- ZDE je místo vlastní funkci !!!!
# funkce reprezentující výpočet neuronové funkce
# funkce dostane na vstupu vstupy neuronové sítě inp, a váhy hran wei
# vrátí seznam hodnot výstupních neuronů

# Activation function
def relu(x):
    return 1 / (1 + np.exp(-x))

# Neural network function
# Outputs: index of the highest value in the output layer
# The number of inputs is equal to the number of sensors
# The number of outputs is equal to the number of possible actions
# The number of hidden layers and neurons is arbitrary
# The number of weights is equal to the number of connections between neurons
def nn_function(inputs, weights):
    num_inputs = len(inputs)
    num_hidden1 = 9
    num_hidden2 = 9  # new hidden layer
    num_outputs = 7
    hidden_weights1 = np.reshape(weights[:num_inputs * num_hidden1], (num_hidden1, num_inputs))
    hidden_weights2 = np.reshape(weights[num_inputs * num_hidden1:num_inputs * num_hidden1 + num_hidden1 * num_hidden2], (num_hidden2, num_hidden1))  # weights for new hidden layer
    output_weights = np.reshape(weights[num_inputs * num_hidden1 + num_hidden1 * num_hidden2:], (num_outputs, num_hidden2))
    hidden_layer1 = np.array([relu(x) for x in np.dot(hidden_weights1, inputs)])
    hidden_layer2 = np.array([relu(x) for x in np.dot(hidden_weights2, hidden_layer1)])  # new hidden layer
    output_layer = np.dot(output_weights, hidden_layer2)
    return np.argmax(output_layer)


# naviguje jedince pomocí neuronové sítě a jeho vlastní sekvence v něm schované
def nn_navigate_me(me, inp):
    # print(dir(me))
    weights = me.sequence
    ind = nn_function(inp[-1], weights)

    # nahoru, pokud není zeď
    if ind == 0 and me.rect.y - ME_VELOCITY > 0:
        me.rect.y -= ME_VELOCITY
        me.dist += ME_VELOCITY

    # dolu, pokud není zeď
    if ind == 1 and me.rect.y + me.rect.height + ME_VELOCITY < HEIGHT:
        me.rect.y += ME_VELOCITY
        me.dist += ME_VELOCITY

    # doleva, pokud není zeď
    if ind == 2 and me.rect.x - ME_VELOCITY > 0:
        me.rect.x -= ME_VELOCITY
        me.dist += ME_VELOCITY

    # doprava, pokud není zeď    
    if ind == 3 and me.rect.x + me.rect.width + ME_VELOCITY < WIDTH:
        me.rect.x += ME_VELOCITY
        me.dist += ME_VELOCITY


# updatuje, zda me vyhrali 
def check_mes_won(mes, flag):
    for me in mes:
        if me.alive and not me.won:
            if me_won(me, flag):
                me.won = True



# resi pohyb mes
def handle_mes_movement(mes, mines, flag):
    for me in mes:
        if me.alive and not me.won:
            inp = [senzor(me, mines, flag)]
            nn_navigate_me(me, inp)

# updatuje timery jedinců
def update_mes_timers(mes, timer):
    [setattr(me, 'timealive', timer) for me in mes if me.alive and not me.won]

# ---------------------------------------------------------------------------
# fitness funkce výpočty jednotlivců
#----------------------------------------------------------------------------

# funkce pro výpočet fitness všech jedinců
WIN_REWARD = 100000
DEATH_PENALTY = 10000
TIME_ALIVE_BONUS = 15
MOVEMENT_PENALTY = 50
MOVEMENT_REWARD = 25
FLAG_DISTANCE_REWARD = 100000

# Update the agent's position
# Save the last position and check if the agent is moving
# If the agent is not moving, increase the time still counter and do not update the last position
# If the agent is moving, reset the time still counter
# If the agent is moving, update the last position
def update(self):
    # Pokud se hybnul
    if (self.rect.x, self.rect.y) == self.last_position:
        self.is_moving = False
        self.time_still += 1
    else:
        self.is_moving = True
        self.time_still = 0
        self.last_position = (self.rect.x, self.rect.y)

# Gaussian mutation with a probability of indpb
# The mutation strength is determined by the sigma parameter
def mutGaussian(individual, indpb, mu=0, sigma=0.2):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] += random.gauss(mu, sigma)
            individual[i] = min(max(0.0, individual[i]), 1.0)
    return individual,


# Calculate the fitness of all agents
# The fitness is calculated based on the distance to the flag, the time alive, the time still, and the movement direction
# The fitness is higher if the agent is closer to the flag, alive, moving towards the flag, and has been alive for a longer time
# The fitness is lower if the agent is further from the flag, dead, moving away from the flag, and has been still for a longer time
# The fitness is zero if the agent has won
# The fitness is negative if the agent has died
def handle_mes_fitnesses(mes, flag):
    for me in mes:
        dx = flag.rect.x - me.rect.x
        dy = flag.rect.y - me.rect.y
        distance_to_flag = dx**2 + dy**2
        if me.won:
            me.fitness = WIN_REWARD - me.timealive
        elif not me.alive:
            me.fitness = me.dist - DEATH_PENALTY + me.timealive * TIME_ALIVE_BONUS + FLAG_DISTANCE_REWARD / distance_to_flag
            if not me.is_moving:
                me.fitness -= MOVEMENT_PENALTY * me.time_still
        else:
            # Reward for moving towards the flag and penalize for moving away from it
            if dx * me.rect.x + dy * me.rect.y > 0:
                me.fitness += MOVEMENT_REWARD
            else:
                me.fitness -= MOVEMENT_PENALTY


# uloží do hof jedince s nejlepší fitness
def update_hof(hof, mes):
    l = [me.fitness for me in mes]
    ind = np.argmax(l)
    hof.sequence = mes[ind].sequence.copy()


def genetic_algorithm(population, toolbox, cxpb, mutpb, ngen):
    # Evaluate the entire population
    # The fitness is calculated based on the distance to the flag, the time alive, the time still, and the movement direction
    for gen in range(ngen):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)

        # Apply mutation on the offspring
        # The mutation strength is determined by the indpb parameter
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)

        population[:] = offspring

# ----------------------------------------------------------------------------
# main loop 
# ----------------------------------------------------------------------------

def main():
    # =====================================================================
    # <----- ZDE Parametry nastavení evoluce !!!!!
    VELIKOST_POPULACE = 100
    EVO_STEPS = 5  # pocet kroku evoluce
    DELKA_JEDINCE =  11*9 + 9*9 + 9*7   # <--------- záleží na počtu vah a prahů u neuronů !!!!!
    NGEN = 30        # počet generací
    CXPB = 0.6          # pravděpodobnost crossoveru na páru
    MUTPB = 0.2        # pravděpodobnost mutace

    SIMSTEPS = 1000

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_rand", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_rand, DELKA_JEDINCE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mutate", mutGaussian, indpb=0.07)

    # vlastni random mutace

    # Random mutation
    # The mutation strength is determined by the mutation_strength parameter
    # The mutation probability is determined by the indpb parameter
    # The mutation is applied to each gene with a probability of indpb
    # The mutation strength is a random number between -mutation_strength and mutation_strength
    def mutRandom(individual, indpb, mutation_strength=0.2):
        for i in range(len(individual)):
            # Apply mutation with a probability of indpb
            if random.random() < indpb:
                # random number between -1 and 1
                # random number between -mutation_strength and mutation_strength
                change = mutation_strength * (random.random() - 0.5) * 2
                new_value = individual[i] + change
                # Limit the new value to the allowed range
                # The new value is limited to the range between 0 and 1
                individual[i] = min(max(0.0, new_value), 1.0)
        return individual,

    toolbox.register("mutate", mutRandom, indpb=0.1, mutation_strength=0.2)
    toolbox.register("mate", tools.cxBlend, alpha=0.6)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("selectbest", tools.selBest)


    # inicializace populace 
    pop = toolbox.population(n=VELIKOST_POPULACE)

    # =====================================================================
    clock = pygame.time.Clock()
    # =====================================================================
    # testování hraním a z toho odvození fitness 
    mines = []
    mes = set_mes(VELIKOST_POPULACE)
    flag = Flag()

    hof = Hof()
    run = True

    level = 3  # <--- ZDE nastavení obtížnosti počtu min !!!!!
    generation = 0

    evolving = True
    evolving2 = False
    timer = 0

    while run:

        clock.tick(FPS)
        # pokud evolvujeme pripravime na dalsi sadu testovani - zrestartujeme scenu
        if evolving:
            timer = 0
            generation += 1
            reset_mes(mes, pop) # přiřadí sekvence z populace jedincům a dá je na start !!!!
            mines = set_mines(level)
            evolving = False

        timer += 1
        check_mes_won(mes, flag)
        handle_mes_movement(mes, mines, flag)

        handle_mines_movement(mines)

        mes_collision(mes, mines)

        if all_dead(mes):
            evolving = True

        update_mes_timers(mes, timer)
        draw_window(mes, mines, flag, level, generation, timer)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # ---------------------------------------------------------------------
        # <---- ZDE druhá část evoluce po simulaci  !!!!!
        # TODO rewrite it and improve it 
        # druhá část evoluce po simulaci, když všichni dohrají, simulace končí 1000 krocích
        if timer >= SIMSTEPS or nobodys_playing(mes):

            # přepočítání fitness funkcí, dle dat uložených v jedinci
            # každý me má svou fitness, která je uložena v jeho objektu
            # každý me odpovídá jednomu jedinci v populaci
            # každý jedinec má svou fitness, která je uložena v jeho objektu
            # každý jedinec odpovídá jednomu me
            handle_mes_fitnesses(mes, flag)   # <--------- ZDE funkce výpočtu fitness !!!!

            update_hof(hof, mes)
            #plot fitnes funkcí
            ff = [me.fitness for me in mes]
            print(ff)

            # přiřazení fitnessů z jedinců do populace
            # každý me si drží svou fitness, a každý me odpovídá jednomu jedinci v populaci
            for i in range(len(pop)):
                ind = pop[i]
                me = mes[i]
                ind.fitness.values = (me.fitness, )


            # evoluce populace
            # výběr nejlepších jedinců
            # křížení a mutace
            # nová populace
            genetic_algorithm(pop, toolbox, CXPB, MUTPB, NGEN)


            evolving = True
    # po vyskočení z cyklu aplikace vytiskne DNA sekvecni jedince s nejlepší fitness
    # a ukončí se

    pygame.quit()


if __name__ == "__main__":
    main()
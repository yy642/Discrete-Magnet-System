import numpy as np
import sys
import time
import numpy.random as rn
from utils import *
from analyze import *

def write(filemane, strs):
    f=open(filename, 'a')
    for s in strs:
        f.write(s)
    f.close()

def initial_temp(state,ac_ratio=0.8):
    cost = cost_function(state)
    eq_steps = 100*total_magnets
    temp=3000
    while True:
        ac_count = 0
        for step in range(eq_steps):
            flip_idx = random_neighbour(state)

            for idx in flip_idx:
                state[idx] = -state[idx]
            new_cost = cost_function(state)
            if acceptance_probability(cost, new_cost, temp) > rn.random():
                cost = new_cost
                ac_count += 1
            else:
                """unflip the magnets"""
                for idx in flip_idx:
                    state[idx] = -state[idx]
        cur_ratio =  ac_count / eq_steps
        if  cur_ratio < ac_ratio:
            temp *= 2
        elif cur_ratio > ac_ratio * 1.05:
            temp /= 1.1
        else:
            return temp

def temperature(step, T0, alpha):
    """ Example of temperature dicreasing as the process goes on."""
    return T0 * alpha ** step

def acceptance_probability(cost, new_cost, temperature):
    if new_cost < cost:
        return 1
    else:
        p = np.exp(- (new_cost - cost) / temperature)
        return p

def initial_guess():
    """ initial guess """
    return np.random.randint(2, size=total_magnets) * 2 - 1

def perfect_pairing_initial_guess():
    """ initial guess """
    pads = np.ones([total_magnets])
    pads[:len(pads)//4]=-1
    np.random.shuffle(pads[:len(pads)//2])
    np.random.shuffle(pads[len(pads)//2:])
    return pads


def random_neighbour(x):
    """random flip one magnets"""
    return [rn.randint(total_magnets), rn.randint(total_magnets)]

def cost_function(x):
    """ Cost of x = f(x)."""
    return f(x)

def f(x):
    """ 
    Function to minimize.
    TEs: a list of energy tensor 
    """
    return np.sum((Compute(x[:half_magnets],TFs, x[half_magnets:])) ** 2) + weight * np.sum(Compute(x[:half_magnets],TEs, x[half_magnets:]))

def check(x):
    pes = Compute(x[:half_magnets],T, x[half_magnets:])
    ex = extrema_info(pes,dlist) 
    if len(ex) < 3:
        return False
    match_z = len(ex) == 3 and abs(ex[0][0] - zlist[0])<=0.1 and abs(ex[2][0] - zlist[1])<=0.1
    diff = (ex[1][1] - ex[2][1] >= 0.1) and (ex[1][1] - ex[0][1] >= 0.1)
    match_e = match_z and (ex[2][1] < elist[1] or ex[0][1] < elist[0])
    if match_z and diff and ex not in seen:
        if match_e:
            write(filename, ["! "+str(ex)+"\n", repr(x.astype('int8')) + "\n"])
        else:
            write(filename, [str(ex)+"\n", repr(x.astype('int8')) + "\n"])
        seen.add(ex)
    #res = match_e and diff 
    return False
    

def annealing(initial_guess,
              cost_function,
              random_neighbour,
              acceptance_probability,
              temperature,
              check,
              random_seed,
              max_initial_states = 10,
              debug=True,
              T0=1000, 
              alpha=0.1):
    """ Optimize the black-box function 'cost_function' with the simulated annealing algorithm."""
   
    """
    generate initial state
    """
    np.random.seed(random_seed)
    state = initial_guess()

    """
    initialize cost based on initial state
    """
    success=False
    cost = cost_function(state)
    T0 = initial_temp(state) 
    temp = temperature(0, T0, alpha)

    initial_states = [] #keep a list of ten recent candidates to restart with
    eq_steps = 100 * total_magnets
    sweeps = total_magnets
    ac_count = 0
    for step in range(eq_steps):               
        flip_idx = random_neighbour(state)
        for idx in flip_idx:
            state[idx] = -state[idx]
        new_cost = cost_function(state)
        if acceptance_probability(cost, new_cost, temp) > rn.random():
            cost = new_cost
            ac_count += 1
            if len(initial_states) > max_initial_states:
                initial_states = initial_states[-max_initial_states:]
            if ac_count % 10 == 0:
                initial_states.append(state)
        else:
            """unflip the magnets"""
            for idx in flip_idx:
                state[idx] = -state[idx]
                
    ac_rate = ac_count / eq_steps
    write(filename, ["T0=" + str(int(T0))+ ", equilibrium for "+ str(eq_steps) + " steps, accept rate = " + str(np.round(ac_rate,2)) +", number of initial guess=" + str(len(initial_states)) + "\n"])

    best_cost = cost
    best_state = state 
    final_step=0

    for i in range(len(initial_states)):
        write(filename, ["initial state:" + str(i)+"\n"])
        initial_state = initial_states[i]
        cost = cost_function(initial_state) 
        temp = temperature(0, T0, alpha)
        step = 1 
        ac_rates = [ac_rate]
        while True:
            temp = temperature(step, T0, alpha)
            ac_count = 0
            for inner_step in range(sweeps):
                flip_idx = random_neighbour(state)
            
                for idx in flip_idx:
                    state[idx] = -state[idx]
                new_cost = cost_function(state)
                
                rn_prob = rn.random()
                if acceptance_probability(cost, new_cost, temp) >  rn_prob:
                    cost = new_cost
                    ac_count += 1
                else:
                    """unflip the magnets"""
                    for idx in flip_idx:
                        state[idx] = -state[idx]
    
            ac_rate = ac_count / sweeps
            ac_rates.append(ac_rate)
            if check(state):
                best_state=state
                success = True
                break

            if len(ac_rates) > 100 and np.sum(ac_rates[-20:]) == 0.0:
                break
                    
            step += 1
            final_step += 1
        if success:
            break
            
    return best_state, cost, success, final_step

if __name__ == "__main__":    
    """
    read in parameters
    """    
    alpha=float(sys.argv[1])
    weight=float(sys.argv[2])
    filename=sys.argv[3]
    
    """
    define target
    """
    zlist=[1.2,4.0]
    elist=[-0.50,-0.50]
    
    """
    define search space
    """
    N=6
    total_magnets=int(2*N*N)
    half_magnets=int(N*N)
    
    TFs=face_to_face_force_tensor(gen_2D_pos(N),zlist)
    TEs=face_to_face_tensor(gen_2D_pos(N),zlist)

    dlist=np.arange(0.8,5,0.1)
    T=face_to_face_tensor(gen_2D_pos(N),dlist)

    random_seeds=np.arange(1)
    write(filename, ["N="+str(N)+", zlist="+str(zlist)+"\n","elist="+str(elist)+"\n", "weight=" + str(weight) +", alpha="+str(alpha)+ ", auto-select T0, random seeds=" +str(random_seeds) + "\n"])
    seen=set()
    for random_seed in random_seeds:
        write(filename, ["random seed=" +str(random_seed) + "\n"])
        start=time.time()
        state, cost, success, final_step = annealing(perfect_pairing_initial_guess, 
                                                     cost_function, 
                                                     random_neighbour, 
                                                     acceptance_probability, 
                                                     temperature,
                                                     check,
                                                     random_seed,
                                                     debug=False,
                                                     alpha=alpha);
        
        end=time.time()
        if not success:
            write(filename, ["fail, total time=" + str(end-start)+ ", number of steps=" + str(final_step)+ ", cost="+ str(cost) + "\n"])
        else:
            write(filename, ["success, total time=" + str(end-start), ", number of steps=" + str(final_step)+ ", cost="+ str(cost) + "\n" + repr(state.astype('int8')) + "\n"])

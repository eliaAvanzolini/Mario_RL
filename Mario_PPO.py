import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as _mp
from torch.distributions import Categorical

from PPO.env import create_train_env, MultipleEnvironments
from PPO.model import PPO
from PPO.process import eval
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

# Scegli la modalità: 'train' o 'test'
MODE = 'test'

# Parametri di Ambiente e Output
WORLD = 1
STAGE = 1
ACTION_TYPE = "simple"  # Opzioni: "right", "simple", "complex"
SAVED_PATH = "trained_models"
OUTPUT_PATH = "output"
LOG_PATH = "tensorboard/ppo_super_mario_bros"


LR = 1e-4             # Tasso di apprendimento
GAMMA = 0.9           # Fattore di sconto (discount factor)
TAU = 1.0             # Parametro per GAE (Generalized Advantage Estimation)
BETA = 0.01           # Coefficiente di Entropia
EPSILON = 0.2         # Parametro di Clipping per PPO
BATCH_SIZE = 16       # Dimensione del mini-batch per l'aggiornamento
NUM_EPOCHS = 10       # Numero di epoche di aggiornamento per ogni raccolta di dati
NUM_LOCAL_STEPS = 512 # Numero di passi raccolti da ogni ambiente prima di un aggiornamento
NUM_GLOBAL_STEPS = 5e6# Numero totale di passi
NUM_PROCESSES = 8     # Numero di ambienti paralleli (parallel workers)
SAVE_INTERVAL = 50    # Intervallo di salvataggio (in episodi)
MAX_ACTIONS = 200


def select_actions(action_type_str):
    if action_type_str == "right":
        return RIGHT_ONLY
    elif action_type_str == "simple":
        return SIMPLE_MOVEMENT
    else:
        return COMPLEX_MOVEMENT

def train_ppo():
    # Inizializzazione e setup
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
        
    if os.path.isdir(LOG_PATH):
        shutil.rmtree(LOG_PATH)
    os.makedirs(LOG_PATH)
    if not os.path.isdir(SAVED_PATH):
        os.makedirs(SAVED_PATH)
        
    # Setup del multiprocessing per l'esecuzione parallela
    mp = _mp.get_context("spawn")
    
    # Creazione degli ambienti multipli
    envs = MultipleEnvironments(WORLD, STAGE, ACTION_TYPE, NUM_PROCESSES)
    
    # Inizializzazione del modello PPO (Actor-Critic)
    model = PPO(envs.num_states, envs.num_actions) 
    
    if torch.cuda.is_available():
        model.cuda()
    
    # Abilita la condivisione della memoria tra i processi per il modello
    model.share_memory()
    
    # Processo di valutazione separato
    process = mp.Process(target=eval, args=((WORLD, STAGE, ACTION_TYPE, MAX_ACTIONS, SAVED_PATH, NUM_PROCESSES, LOG_PATH), model, envs.num_states, envs.num_actions))
    process.start()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Preparazione degli stati iniziali
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()
        
    curr_episode = 0
    
    print(f"Avvio addestramento PPO su {NUM_PROCESSES} processi paralleli.")

    while True:
        curr_episode += 1
        
        # 1. Raccolta delle Traiettorie (Rollout)
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []
        
        for _ in range(NUM_LOCAL_STEPS):
            states.append(curr_states)
            logits, value = model(curr_states)
            values.append(value.squeeze())
            
            # Policy e Campionamento dell'azione
            policy = F.softmax(logits, dim=1)
            old_m = Categorical(policy)
            action = old_m.sample()
            actions.append(action)
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)
            
            # Esecuzione dell'azione negli ambienti paralleli
            if torch.cuda.is_available():
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
            else:
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]

            # Ricezione di stato, ricompensa, done
            state, reward, done, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            
            # Gestione e formattazione dei dati
            state = torch.from_numpy(np.concatenate(state, 0))
            if torch.cuda.is_available():
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done = torch.cuda.FloatTensor(done)
            else:
                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)
                
            rewards.append(reward)
            dones.append(done)
            curr_states = state

        # 2. Calcolo dei Ritorni (Returns) e del Vantaggio (Advantage - GAE)
        _, next_value, = model(curr_states)
        next_value = next_value.squeeze()
        
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        
        gae = 0
        R = [] # Lista per i Ritorni (Estimated Returns)
        
        # Calcolo dei ritorni e GAE (Generalized Advantage Estimation) 
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            # Aggiornamento GAE
            gae = gae * GAMMA * TAU
            gae = gae + reward + GAMMA * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            R.append(gae + value) # R = Advantage + Value Function (il Ritorno atteso)
            
        R = R[::-1]
        R = torch.cat(R).detach()
        advantages = R - values # Vantaggio = Ritorno - Valore Stimato
        
        # 3. Aggiornamento della Policy (Mini-Batch Gradient Descent)
        for i in range(NUM_EPOCHS):
            total_samples = NUM_LOCAL_STEPS * NUM_PROCESSES
            indice = torch.randperm(total_samples)
            
            # Loop per mini-batch
            for j in range(BATCH_SIZE):
                batch_start = int(j * (total_samples / BATCH_SIZE))
                batch_end = int((j + 1) * (total_samples / BATCH_SIZE))
                batch_indices = indice[batch_start:batch_end]
                
                # Forward Pass
                logits, value = model(states[batch_indices])
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                
                # Calcolo del Rapporto (Ratio) e del Clipping (Policy Loss)
                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                
                # Funzione Obiettivo Clipped di PPO (Actor Loss) 
                actor_loss = -torch.mean(torch.min(
                    ratio * advantages[batch_indices],
                    torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * advantages[batch_indices]))
                    
                # Loss della Funzione Valore (Critic Loss)
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                
                # Loss di Entropia (per incoraggiare l'esplorazione)
                entropy_loss = torch.mean(new_m.entropy())
                
                # Loss Totale
                total_loss = actor_loss + critic_loss - BETA * entropy_loss
                
                # Ottimizzazione
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Clipping del gradiente per stabilità
                optimizer.step()
                
        # Stampa e Salvataggio (la logica di salvataggio è commentata nello script originale)
        if curr_episode % SAVE_INTERVAL == 0 and curr_episode > 0:
             torch.save(model.state_dict(),
                        f"{SAVED_PATH}/ppo_super_mario_bros_{WORLD}_{STAGE}")
             
        print(f"Episodio: {curr_episode}. Total loss: {total_loss.item():.4f}")


def test_ppo():
    # Parametro per la fase di test (usato nella funzione eval, ma utile mantenerlo qui)
    MAX_ACTIONS = 200 # Massimo numero di passi prima di resettare in fase di test

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
        
    actions = select_actions(ACTION_TYPE)

    # Creazione dell'ambiente di test (con l'opzione di salvare un video)
    if not os.path.isdir(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        
    env = create_train_env(WORLD, STAGE, actions,
                           f"{OUTPUT_PATH}/video_{WORLD}_{STAGE}.mp4")
                           
    # Inizializzazione del modello
    model = PPO(env.observation_space.shape[0], len(actions))
    
    # Caricamento dei pesi addestrati
    model_path = f"{SAVED_PATH}/ppo_super_mario_bros_{WORLD}_{STAGE}"
    if not os.path.exists(model_path):
        print(f"ERRORE: Modello non trovato in {model_path}. Assicurati di aver eseguito il training.")
        return

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        
    model.eval() # Imposta il modello in modalità valutazione
    
    state = torch.from_numpy(env.reset())
    print(f"Avvio test: World {WORLD}, Stage {STAGE}")
    
    while True:
        if torch.cuda.is_available():
            state = state.cuda()
            
        # Forward Pass: ottiene logit e valore dallo stato
        logits, value = model(state)
        
        # Policy e Scelta dell'azione
        policy = F.softmax(logits, dim=1)
        
        # In fase di test, spesso si prende l'azione con la probabilità massima (determinismo)
        action = torch.argmax(policy).item()
        
        # Esecuzione dell'azione
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        
        # Renderizza il gioco (visibile)
        env.render()
        
        # Verifica del completamento del livello
        if info["flag_get"]:
            print(f"World {WORLD} stage {STAGE} completato con successo!")
            break
            
        if done:
            print(f"Test terminato (Morte o Timeout). World {WORLD}, Stage {STAGE}")
            break
            
    env.close()


if __name__ == "__main__":
    if MODE == 'train':
        train_ppo()
    elif MODE == 'test':
        # Parametro MAX_ACTIONS è necessario per la funzione 'eval' interna al training
        # ma non per la funzione 'test_ppo' qui definita. Lo imposto a 200 per coerenza
        # con il codice originale, anche se non è usato direttamente.
        test_ppo()
    else:
        print("MODALITÀ non valida. Imposta MODE su 'train' o 'test'.")
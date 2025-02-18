import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np

def load_results(filepath):

    data = np.load(filepath)
    return data

def plot_PPs(data, kappa=5.0):

    # Extract PP_t and PP arrays
    PP_t = data['PP_t']  
    PP = data['PP']     

    n_subjects = PP_t.shape[0]
    fig, axs = plt.subplots(1, n_subjects, figsize=(10, 5))

    for i in range(n_subjects):

        mean_PP_t = PP_t[i, 0, 0, :, :].mean(axis=0)
        std_PP_t = PP_t[i, 0, 0, :, :].std(axis=0)
        x_values = np.arange(mean_PP_t.shape[0])
        conf_interval_PP_t = 1.96 * std_PP_t / PP_t.shape[3]
        mean_PP = PP[i, 0, 0, :, :].mean(axis=0)
        std_PP = PP[i, 0, 0, :, :].std(axis=0)
        conf_interval_PP = 1.96 * std_PP / PP.shape[3]

        axs[i].plot(mean_PP_t, label=f'PP_t', color='blue')
        axs[i].plot(mean_PP, label=f'PP', color='red')
        axs[i].fill_between(x_values, mean_PP_t - conf_interval_PP_t, mean_PP_t + conf_interval_PP_t, color="blue", alpha=0.3)
        axs[i].fill_between(x_values, mean_PP - conf_interval_PP, mean_PP + conf_interval_PP, color="red", alpha=0.3)
        ''' 
        axs[i].plot(mean_PP_t, label=f'PP_t', color='blue')
        axs[i].plot(mean_PP, label=f'PP', color='red')
        axs[i].fill_between(x_values, mean_PP_t - 1* std_PP_t, mean_PP_t + 1* std_PP_t,
            color="skyblue", alpha=0.15, label="95% Confidence Interval")
        axs[i].fill_between(x_values, mean_PP - 1* std_PP_t, mean_PP_t + 1* std_PP_t,
                 color="lightsalmon", alpha=0.15, label="95% Confidence Interval")
        '''
        axs[i].set_title(f"Subject {i}")
        axs[i].set_xlabel("Iterations")  # X-axis label for each subplot
        axs[i].set_ylabel("Scores (%)")
        
    fig.suptitle(f"Exploration vs. Exploitation Scores with kappa={kappa}", fontsize=14, fontweight="bold")    
    plt.legend(["Exploitation", "Exploration"], loc="lower right")
    plt.tight_layout()
    plt.show()

def comparing_exploitation(data1, data2, kappa=5.0):
    # Extract PP_t arrays
    PP_t1 = data1['PP_t']
    PP_t2 = data2['PP_t']

    n_subjects = PP_t1.shape[0]
    fig, axs = plt.subplots(1, n_subjects, figsize=(10, 5))

    for i in range(n_subjects):
        mean_PP_t1 = PP_t1[i, 0, 0, :, :].mean(axis=0)
        std_PP_t1 = PP_t1[i, 0, 0, :, :].std(axis=0)
        conf_interval_PP_t1 = 1.96 * std_PP_t1 / np.sqrt(PP_t1.shape[3])

        mean_PP_t2 = PP_t2[i, 0, 0, :, :].mean(axis=0)
        std_PP_t2 = PP_t2[i, 0, 0, :, :].std(axis=0)
        conf_interval_PP_t2 = 1.96 * std_PP_t2 / np.sqrt(PP_t2.shape[3])

        x_values = np.arange(mean_PP_t1.shape[0])

        axs[i].plot(mean_PP_t1, color='blue')
        axs[i].plot(mean_PP_t2, color='green')

        axs[i].fill_between(x_values, mean_PP_t1 - conf_interval_PP_t1, mean_PP_t1 + conf_interval_PP_t1, color="blue", alpha=0.3)
        axs[i].fill_between(x_values, mean_PP_t2 - conf_interval_PP_t2, mean_PP_t2 + conf_interval_PP_t2, color="green", alpha=0.3)

        axs[i].set_title(f"Subject {i}")
        axs[i].set_xlabel("Iterations")
        axs[i].set_ylabel("Scores (%)")

    fig.suptitle(f"Comparing Exploitation Scores with kappa={kappa}", fontsize=14, fontweight="bold")
    plt.legend(["Simple GP", "SpatialTemporo GP"], loc="lower right")
    plt.tight_layout()
    plt.show()

def comparing_exploration(data1, data2, kappa=5.0):
    # Extract PP arrays
    PP1 = data1['PP']
    PP2 = data2['PP']

    n_subjects = PP1.shape[0]
    fig, axs = plt.subplots(1, n_subjects, figsize=(10, 5))

    for i in range(n_subjects):
        mean_PP1 = PP1[i, 0, 0, :, :].mean(axis=0)
        std_PP1 = PP1[i, 0, 0, :, :].std(axis=0)
        conf_interval_PP1 = 1.96 * std_PP1 / np.sqrt(PP1.shape[3])

        mean_PP2 = PP2[i, 0, 0, :, :].mean(axis=0)
        std_PP2 = PP2[i, 0, 0, :, :].std(axis=0)
        conf_interval_PP2 = 1.96 * std_PP2 / np.sqrt(PP2.shape[3])

        x_values = np.arange(mean_PP1.shape[0])

        axs[i].plot(mean_PP1, color='blue')
        axs[i].plot(mean_PP2, color='green')

        axs[i].fill_between(x_values, mean_PP1 - conf_interval_PP1, mean_PP1 + conf_interval_PP1, color="blue", alpha=0.3)
        axs[i].fill_between(x_values, mean_PP2 - conf_interval_PP2, mean_PP2 + conf_interval_PP2, color="green", alpha=0.3)

        axs[i].set_title(f"Subject {i}")
        axs[i].set_xlabel("Iterations")
        axs[i].set_ylabel("Scores (%)")

    fig.suptitle(f"Comparing Exploration Scores with kappa={kappa}", fontsize=14, fontweight="bold")
    plt.legend(["Simple GP", "SpatialTemporo GP"], loc="lower right")
    plt.tight_layout()
    plt.show()

def plot_Q(data, kappa=5.0):

    Q = data['Q']

    n_subjects = Q.shape[0]
    fig, axs = plt.subplots(1, n_subjects, figsize=(10, 5))


    for i in range(n_subjects):

        mean_Q = Q[i].mean(axis=2).squeeze()
        std_Q = Q[i].std(axis=2).squeeze()
        x_values = np.arange(mean_Q.shape[0])

        axs[i].plot(mean_Q, label=f'Q', color='orange')
        axs[i].fill_between(x_values, mean_Q - std_Q, mean_Q + std_Q,
            color="moccasin", alpha=0.2, label="95% Confidence Interval")
        
        axs[i].set_title(f"Subject {i}")
        axs[i].set_xlabel("Iterations")  # X-axis label for each subplot
        axs[i].set_ylabel("Values")
        
    fig.suptitle(f"Query value with kappa={kappa}", fontsize=14, fontweight="bold")    
    plt.tight_layout()
    plt.show()

def plot_training_time(data, kappa=5.0):

    T = data['Train_time'][:, 0, 0, :, :]
    n_subjects = T.shape[0]
    fig, axs = plt.subplots(1, n_subjects, figsize=(10, 5))


    for i in range(n_subjects):

        mean_T = T[i].mean(axis=0)
        std_T = T[i].std(axis=0)
        x_values = np.arange(mean_T.shape[0])

        axs[i].plot(mean_T, label=f'T', color='blue')
        axs[i].fill_between(x_values, mean_T - std_T, mean_T + std_T,
            color="skyblue", alpha=0.2, label="95% Confidence Interval")
        
        axs[i].set_title(f"Subject {i}")
        axs[i].set_xlabel("Iterations")  # X-axis label for each subplot
        axs[i].set_ylabel("Training time ")
        
    fig.suptitle(f"Training time with kappa={kappa}", fontsize=14, fontweight="bold")    
    plt.tight_layout()
    plt.show()

def plot_loss(data, kappa=5.0):
    
    loss = data['LOSS'][:, 0, 0, :, :]
    n_subjects = loss.shape[0]
        
    fig, axs = plt.subplots(1, n_subjects, figsize=(10, 5))

    for i in range(n_subjects):

        l = loss[i].mean(axis=0)

        axs[i].plot(l, label=f'loss', color='blue')
        axs[i].set_title(f"Subject {i}")
        axs[i].set_xlabel("Iterations")  # X-axis label for each subplot
        axs[i].set_ylabel("Loss ")
        
    fig.suptitle(f"Loss curves with kappa={kappa}", fontsize=14, fontweight="bold")    
    plt.tight_layout()
    plt.show()

def plot_loss(data, kappa=5.0):
    loss = data['LOSS'][:, 0, 0, :, :]
    n_subjects = loss.shape[0]

    plt.figure(figsize=(10, 5))

    for i in range(n_subjects):
        l = loss[i].mean(axis=0)
        plt.plot(l, label=f'Subject {i}')  # Different color for each subject

    plt.title(f"Loss curves with kappa={kappa}", fontsize=14, fontweight="bold")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)  # Optional: adds a grid for better readability
    plt.show()




if __name__ == '__main__':

    filepath_vanilla = './output/experiments/VanillaGPBO_250216_4channels_artRej_lr001_5rnd.npz'
    filepath_spatiotemp = './output/experiments/ParallelizedGP_250216_4channels_artRej_lr001_5rnd.npz'
    filepath_parallelized = './output/experiments/TemporoSpatialGP_250214_4channels_artRej_lr001_5rnd.npz'
    data_vanilla = load_results(filepath_vanilla)
    data_spatiotemp = load_results(filepath_spatiotemp)
    data_parallelized = load_results(filepath_parallelized)
    comparing_exploitation(data_vanilla, data_parallelized)
    comparing_exploration(data_vanilla, data_parallelized)
    plot_training_time(data_parallelized)
    plot_training_time(data_spatiotemp)
    plot_training_time(data_vanilla)
    #plot_Q(data)
    #plot_PPs(data_parallelized)
    #plot_loss(data)
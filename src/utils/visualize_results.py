import matplotlib.pyplot as plt
import seaborn as sns
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

    T = data['Train_time']
    print(T.shape)
    n_subjects = T.shape[0]
    fig, axs = plt.subplots(1, n_subjects, figsize=(10, 5))


    for i in range(n_subjects):

        mean_T = T[i].mean(axis=2).squeeze()
        std_T = T[i].std(axis=2).squeeze()
        x_values = np.arange(mean_T.shape[0])

        axs[i].plot(mean_T, label=f'T', color='blue')
        axs[i].fill_between(x_values, mean_T - std_T, mean_T + std_T,
            color="moccasin", alpha=0.2, label="95% Confidence Interval")
        
        axs[i].set_title(f"Subject {i}")
        axs[i].set_xlabel("Iterations")  # X-axis label for each subplot
        axs[i].set_ylabel("Training time ")
        
    fig.suptitle(f"Training time with kappa={kappa}", fontsize=14, fontweight="bold")    
    plt.tight_layout()
    plt.show()



def plot_loss(losses, T):
    
    for model_index in range(losses.shape[1]):
        plt.plot(
            range(T),  # Iterations on the x-axis
            losses[:, model_index],  # Performance on the y-axis
            label=f'Model {model_index + 1}'  # Label for the model
        )
    plt.ylim(-1.0,2.0) 
    # Add labels, title, and legend
    plt.xlabel("Iteration")
    plt.ylabel("Average Loss")
    plt.title("Loss Over Iterations")
    plt.legend()
    plt.show()

def plot_values(predictions, T):
    plt.figure(figsize=(12, 5))
    plt.plot(range(T), predictions, label="Best candidate solution")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Value Over Iterations")
    plt.legend()
    plt.show()

def plot_partition_updates(model_updates, n_models, T):
    # Prepare the data for eventplot
    events = [[] for _ in range(n_models)]  # One list per model
    for t, updates in enumerate(model_updates):
        for model in updates:
            events[model].append(t)

    # Plot the eventplot
    plt.eventplot(events, orientation='horizontal', lineoffsets=range(n_models), linelengths=0.8)

    # Add labels and title
    plt.xlabel("Iteration (t)")
    plt.ylabel("Model Index (k)")
    plt.title("Eventplot of Model Updates")
    plt.yticks(range(n_models), labels=[f"Model {k}" for k in range(n_models)])
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()

def plot_exploration_exploitation(exploration_performance, exploitation_performance, T):
    
    # Plot exploration performance over iterations
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        range(T),  # Iterations on the x-axis
        exploration_performance) # Performance on the y-axis)
    # Add labels, title, and legend
    plt.xlabel('Iteration')
    plt.ylabel('Exploration Performance')
    plt.title('Exploration Score for each Model')
    plt.legend()

    # Plot exploitation performance over iterations
    plt.subplot(1, 2, 2)
    plt.plot(
        range(T),  # Iterations on the x-axis
        exploitation_performance,  # Performance on the y-axis
    )
    plt.xlabel("Iteration")
    plt.ylabel('Exploitation Performance')
    plt.title('Exploitation Score for each Model')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_mae(maes, T):
    for model_index in range(maes.shape[1]):
        plt.plot(
            range(T),  # Iterations on the x-axis
            maes[:, model_index],  # Performance on the y-axis
            label=f'Model {model_index + 1}'  # Label for the model
        )
    plt.xlabel("Iteration")
    plt.ylabel("Mean Abs Error")
    plt.title("MAE Over Iterations")
    plt.legend()
    plt.show()

    
    

if __name__ == '__main__':

    filepath = './output/experiments/TemporoSpatialGP_250211_4channels_artRej_kappa20_lr001_5rnd.npz'
    data = load_results(filepath)
    plot_training_time(data)
    #plot_Q(data)
    #plot_PPs(data)
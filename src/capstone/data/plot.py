from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def plot_simulation_data(df, simulation_run, title=None, fault_number=0):
    process_variables = df.columns[3:]
    df_sim = df[(df.simulationRun == simulation_run)&(df.faultNumber==fault_number)]
    df_melt = pd.melt(df_sim, id_vars=['sample'], value_vars=process_variables, var_name='variable', value_name='value')
    rp = sns.relplot(data=df_melt, x='sample', y='value', col='variable', kind='line', 
                col_wrap=6, height=2,
                 facet_kws={'sharey': False, 'sharex': True})
    if(title is not None):
        rp.fig.suptitle(f'{title}: Simulation #{simulation_run}')

def plot_simulation_variable(df, simulation_run, title, variable, fault_number=0):
    fig, ax = plt.subplots(figsize=(7,4))
    df_sim = df[(df.simulationRun == simulation_run)&(df.faultNumber==fault_number)]
    sns.lineplot(data=df_sim, x='sample', y=variable, ax=ax)
    if(title is not None):
        fig.suptitle(f'{title}: Simulation #{simulation_run} - {variable}')


def plot_correlation_clustermap(df):
    corr = df.corr('spearman')
    sns.clustermap(corr, annot=True, fmt='.2f', figsize=(30,20), cmap='coolwarm', row_cluster=False)

def plot_correlation_matrix(df):
    corr = df.corr('spearman')
    fig, ax = plt.subplots(figsize=(30,20))
    sns.heatmap(corr, annot=True, fmt='.2f', ax=ax, cmap='coolwarm', center=0)
    fig.show()


def plot_encoded_variables(df, df_rec, process_variables, simulation_run, fault_number=0):
    df_sim = df[(df.simulationRun == simulation_run)&(df.faultNumber==fault_number)]
    df_sim_rec = df_rec[(df_rec.simulationRun == simulation_run)&(df_rec.faultNumber==fault_number)]

    df_melt = pd.melt(df_sim, id_vars=['sample'], value_vars=process_variables, var_name='variable', value_name='value')
    df_rec_melt = pd.melt(df_sim_rec, id_vars=['sample'], value_vars=process_variables, var_name='variable', value_name='value')

    df_melt['type'] = 'original'
    df_rec_melt['type'] = 'encoded'

    df_melt = pd.concat([df_melt, df_rec_melt], ignore_index=True) 

    rp = sns.relplot(data=df_melt, x='sample', y='value', hue='type', col='variable', kind='line', 
                col_wrap=6, height=2,
                 facet_kws={'sharey': False, 'sharex': True})


def plot_simulation_anomalies(df, simulation_run, threshold):    
    df_sim = df[(df.simulationRun == simulation_run)]
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df_sim['sample'], df_sim['loss_mae'], label='Absolute Error')
    ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    
    # ax2 = ax.twinx() 
    # rma = df_sim['loss_mae'].rolling(5).mean().diff()
    # ax2.plot(df_sim['sample'], rma, 'r', label='Gradient')

    anomaly_indexes = df_sim[df_sim['loss_mae']>threshold]['sample']
    if(len(anomaly_indexes)>0):
        anomaly_idx = df_sim[df_sim['loss_mae']>threshold]['sample'].values[0]
        ax.axvline(x=anomaly_idx, color='k', linestyle='--', label='Anomaly')
        ax.set_title(f"Anomaly started at sample #{anomaly_idx}")
    else:
        ax.set_title('Anomalies were not detected!')
    
    ax.set_ylabel('Absolute Error')
    ax.set_xlabel('sample #')
    ax.legend()
    fig.show()
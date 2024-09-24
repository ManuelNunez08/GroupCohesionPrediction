
import numpy as np
import pandas as pd
import plotly.graph_objs as go

import plotly.graph_objs as go

config = {
            'staticPlot': True  # Disables zoom, pan, hover, etc.
        }


def get_column(name):
    if name=='General Cohesion':
        return 'Cohesion'
    if name=='Task Cohesion':
        return 'Task'
    if name=='Social Cohesion':
        return 'Social'

def plot_kappa_vs_average_binary(df_kappa, cohesion_type, floor, ceiling, kappa_threshold):
    """
    Plots the relationship between kappa scores and average scores for the specified cohesion type (Task, Social, or Overall).
    Colors points based on whether they are above/below the floor/ceiling and fall within the score range specified.
    Adds legends for 'High Cohesion Observations' and 'Low Cohesion Observations'.
    Returns Plotly-compatible data.
    """

    column_name = get_column(cohesion_type)
    # Define the column names based on the cohesion type
    avg_col = f'{column_name}_Average'
    kappa_col = f'{column_name}_Kappa'

    # Check if the specified cohesion type exists
    if avg_col not in df_kappa.columns or kappa_col not in df_kappa.columns:
        print(f"Invalid cohesion type: {column_name}. Choose 'Task', 'Social', or 'Cohesion'.")
        return []
    
    # Apply threshold conditions
    within_threshold = ((df_kappa[avg_col] <= ceiling) | (df_kappa[avg_col] > floor)) & (df_kappa[kappa_col] >= kappa_threshold)
    high_cohesion = (df_kappa[avg_col] > floor) & (df_kappa[kappa_col] >= kappa_threshold)
    low_cohesion = (df_kappa[avg_col] <= ceiling) & (df_kappa[kappa_col] >= kappa_threshold)
    
    # Create the traces for high and low cohesion observations
    trace_high_cohesion = go.Scatter(
        x=df_kappa[kappa_col][high_cohesion],
        y=df_kappa[avg_col][high_cohesion],
        mode='markers',
        marker=dict(color='green'),
        name=f'High Cohesion Observations: {len(df_kappa[kappa_col][high_cohesion])}'
    )

    trace_low_cohesion = go.Scatter(
        x=df_kappa[kappa_col][low_cohesion],
        y=df_kappa[avg_col][low_cohesion],
        mode='markers',
        marker=dict(color='purple'),
        name=f'Low Cohesion Observations: {len(df_kappa[kappa_col][low_cohesion])}'
    )

    # trace_within = go.Scatter(
    #     x=df_kappa[kappa_col][within_threshold],
    #     y=df_kappa[avg_col][within_threshold],
    #     mode='markers',
    #     marker=dict(color='red'),
    #     name='Within threshold'
    # )

    trace_outside = go.Scatter(
        x=df_kappa[kappa_col][~within_threshold],
        y=df_kappa[avg_col][~within_threshold],
        mode='markers',
        marker=dict(color='blue'),
        name= f'Outside threshold: {len(df_kappa[kappa_col][~within_threshold])}'
    )

    # Layout for the Plotly graph
    layout = go.Layout(
        title=f'{cohesion_type} : Average vs Kappa',
        xaxis=dict(title='Kappa Score'),
        yaxis=dict(title='Average Score'),
        showlegend=True
    )

    # Return traces and layout for Plotly rendering
    return [trace_high_cohesion, trace_low_cohesion, trace_outside], layout, config




def plot_std_vs_mean_binary(df, question_name, jitter_strength=0.00, floor=5, ceiling=3, max_std=1.0):
    mean_col = question_name + '_mean'
    std_col = question_name + '_std'
    
    if mean_col in df.columns and std_col in df.columns:
        means = df[mean_col]
        stds = df[std_col]
        
        # Apply jitter by adding random noise
        jitter_means = means + np.random.normal(0, jitter_strength, size=len(means))
        jitter_stds = stds + np.random.normal(0, jitter_strength, size=len(stds))
        
        within_threshold = ((jitter_means >= floor) | (jitter_means <= ceiling)) & (jitter_stds <= max_std)
        high_cohesion = (jitter_means > floor)  & (jitter_stds <= max_std)
        low_cohesion = (jitter_means <= ceiling) & (jitter_stds <= max_std)
        
        # Plot data using Plotly
        trace_high_cohesion = go.Scatter(
            y=jitter_means[high_cohesion],
            x=jitter_stds[high_cohesion],
            mode='markers',
            marker=dict(color='green'),
            name=f'High Cohesion Observations: {len(jitter_means[high_cohesion])}'
        )
        
        trace_low_cohesion = go.Scatter(
            y=jitter_means[low_cohesion],
            x=jitter_stds[low_cohesion],
            mode='markers',
            marker=dict(color='red'),
            name= f'Low Cohesion Observations: {len(jitter_means[low_cohesion])}'
        )
        
        trace_outside = go.Scatter(
            y=jitter_means[~within_threshold],
            x=jitter_stds[~within_threshold],
            mode='markers',
            marker=dict(color='blue'),
            name= f'Outside thresholds: {len(jitter_means[~within_threshold])}'
        )
        
        
        layout = go.Layout(
            title=f'Question {question_name}: Standard Deviation vs Mean',
            yaxis=dict(title='Mean'),
            xaxis=dict(title='Standard Deviation'),
            showlegend=True
        )
        
        return [trace_high_cohesion, trace_low_cohesion, trace_outside], layout, config 
    else:
        return []



def plot_kappa_vs_average_regress(df_kappa, cohesion_type, floor, ceiling, kappa_threshold):
    """
    Plots the relationship between kappa scores and average scores for the specified cohesion type (Task, Social, or Overall).
    Colors points based on whether they are above the kappa threshold and fall within the score range specified by the floor and ceiling.
    Returns Plotly-compatible data.
    """

    column_name = get_column(cohesion_type)

    # Helper function to apply conditions for coloring
    def apply_thresholds(average, kappa):
        return (((average <= ceiling) | (average >= floor)) & (kappa >= kappa_threshold))

    # Define the column names based on the cohesion type
    avg_col = f'{column_name}_Average'
    kappa_col = f'{column_name}_Kappa'

    # Check if the specified cohesion type exists
    if avg_col not in df_kappa.columns or kappa_col not in df_kappa.columns:
        print(f"Invalid cohesion type: {column_name}. Choose 'Task', 'Social', or 'Cohesion'.")
        return []
    
    # Apply threshold conditions
    within_threshold = apply_thresholds(df_kappa[avg_col], df_kappa[kappa_col])

    # Create the traces for Plotly plot
    trace_within = go.Scatter(
        x=df_kappa[kappa_col][within_threshold],
        y=df_kappa[avg_col][within_threshold],
        mode='markers',
        marker=dict(color='red'),
        name= f'Within threshold: {len(df_kappa[kappa_col][within_threshold])}'
    )

    trace_outside = go.Scatter(
        x=df_kappa[kappa_col][~within_threshold],
        y=df_kappa[avg_col][~within_threshold],
        mode='markers',
        marker=dict(color='blue'),
        name= f'Outside threshold: {len(df_kappa[kappa_col][~within_threshold])}'
    )

    # Layout for the Plotly graph
    layout = go.Layout(
        title=f'{cohesion_type} : Average vs Kappa',
        xaxis=dict(title='Kappa Score'),
        yaxis=dict(title='Average Score'),
        showlegend=True
    )

    # Return traces and layout for Plotly rendering
    return [trace_within, trace_outside], layout, config



def plot_std_vs_mean_regress(df, question_name, jitter_strength=0.00, floor=5, ceiling=3, max_std=1.0):
    mean_col = question_name + '_mean'
    std_col = question_name + '_std'
    
    if mean_col in df.columns and std_col in df.columns:
        means = df[mean_col]
        stds = df[std_col]
        
        # Apply jitter by adding random noise
        jitter_means = means + np.random.normal(0, jitter_strength, size=len(means))
        jitter_stds = stds + np.random.normal(0, jitter_strength, size=len(stds))
        
        within_threshold = ((jitter_means >= floor) | (jitter_means <= ceiling)) & (jitter_stds <= max_std)
        
        # Plot data using Plotly
        trace_outside = go.Scatter(
            y=jitter_means[~within_threshold],
            x=jitter_stds[~within_threshold],
            mode='markers',
            marker=dict(color='blue'),
            name= f'Outside thresholds: {len(jitter_means[~within_threshold])}'
        )
        
        trace_within = go.Scatter(
            y=jitter_means[within_threshold],
            x=jitter_stds[within_threshold],
            mode='markers',
            marker=dict(color='red'),
            name= f'Within thresholds: {len(jitter_means[within_threshold])}'
        )
        
        layout = go.Layout(
            title=f'Question {question_name}: Standard Deviation vs Mean',
            yaxis=dict(title='Mean'),
            xaxis=dict(title='Standard Deviation'),
            showlegend=True
        )
        
        return [trace_outside, trace_within], layout, config
    else:
        return []


# get plot object 
def plot_suite_data_plotly(train_val_targets, suite):
    # Create the histogram using Plotly
    hist_data = go.Histogram(
        x=train_val_targets,
        nbinsx=20,
        marker=dict(color='rgba(55, 128, 191, 0.7)', line=dict(color='black', width=1.5)),
        opacity=0.7
    )

    layout = go.Layout(
        title=f'Distribution of {suite} Data',
        xaxis_title='Target Value (y)',
        yaxis_title='Frequency',
        showlegend=False,
        bargap=0.1,
        margin=dict(l=40, r=40, t=40, b=40),
        autosize=False,
        width=350,
        height=200
    )

    # Disable user interaction
    config = {
        'staticPlot': True,  # Disable user interaction
        'displayModeBar': False  # Disable mode bar (toolbar) for plotly
    }
    
    return [hist_data], layout, config

# for data handling 
import pandas as pd
import pickle

# for data set 
from torch_geometric.data import Data, Dataset

# for model architetcture 
import torch

from torch.utils.data import random_split
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# to save images
from io import BytesIO
import base64

# we use dense weight to balance data set
from model_scripts.optimize_alpha import find_optimal_alpha
from denseweight import DenseWeight


# ======================================= BELOW WE DEFINE THE MODEL DATA SET =========================
class CohesionGraphDataset(Dataset):
    def __init__(self, data_list):
        super(CohesionGraphDataset, self).__init__()
        self.data_list = []

        for entry in data_list:
            # Process node features
            node_features_dict = entry['features'][0]
            node_names = list(node_features_dict.keys())  
            node_features = []

            # convert features to a tensor
            for node in node_names:
                node_data = node_features_dict[node]
                node_features.append([feature[1] for feature in node_data])  

            x = torch.tensor(node_features, dtype=torch.float)

            # Process edges
            edge_features_dict = entry['features'][1]
            edge_index = []
            edge_attr = []

            # Convert into tensors
            for edge, edge_data in edge_features_dict.items():
                src, dst = edge.split(',')
                src_idx = node_names.index(src)  # Get the node index for source
                dst_idx = node_names.index(dst)  # Get the node index for destination

                # Append the edge index 
                edge_index.append([src_idx, dst_idx])

                # Append the edge attributes (features for this edge)
                edge_attr.append([feature[1] for feature in edge_data])

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  
            edge_attr = torch.tensor(edge_attr, dtype=torch.float) 

            # Store the score as the y label
            y = torch.tensor([entry['score']], dtype=torch.float)

            # Create Geometric Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    



# =================== BELOW WE READ IN ALL DATA FROM PICKLES ==============
# Load the graphs data from file
with open('../GNNs/data/annotations_graphs_new.pkl', 'rb') as f:
    graphs_cohesion_data = pickle.load(f)

# Load question-level scores df 
question_level_df = pd.read_pickle("../Cohesion_Annotations/Question_Split_data.pkl")
category_level_df = pd.read_pickle("../Cohesion_Annotations/Cohesion_split_data.pkl")

# Filter df to only include rows where meetings and egment ranges are found in graphs_cohesion_data
graph_meetings_and_starts = {(graph_entry['meeting'], graph_entry['start']) for graph_entry in graphs_cohesion_data}
question_level_df = question_level_df[
    question_level_df.apply(lambda row: (row['Meeting'], row['Start'] * 60) in graph_meetings_and_starts, axis=1)
]
category_level_df = category_level_df[
    category_level_df.apply(lambda row: (row['Meeting'], row['Start'] * 60) in graph_meetings_and_starts, axis=1)
]


def get_column(name):
    if name=='General Cohesion':
        return 'Cohesion'
    if name=='Task Cohesion':
        return 'Task'
    if name=='Social Cohesion':
        return 'Social'
    

# ===================== THE TWO FUNCTIONS BELOW ARE TASKED WITH FILTERING THE DATA AND ASSIGNING LABELS (Binary) ===================

def filter_and_associate_data_question_binary(graph_data, df, question, max_std_dev=1.0, floor=4.5, ceiling=3.5):

    # Filter DataFrame for entries where the standard deviation is below the max_std_dev
    filtered_df = df[(df[f'{question}_std'] <= max_std_dev) & 
                ((df[f'{question}_mean'] <= ceiling) | (df[f'{question}_mean'] >= floor))]
    filtered_graph_data = []

    # Iterate over each graph entry and find corresponding rows in the DataFrame
    for graph_entry in graph_data:
        meeting = graph_entry['meeting']
        start = graph_entry['start']

        # Find the matching row in the filtered DataFrame
        match = filtered_df[(filtered_df['Meeting'] == meeting) & (filtered_df['Start'] * 60 == start)]
        if not match.empty:
            average_score = match[f'{question}_mean'].values[0]
            graph_entry['score'] = 1 if average_score >= floor else 0
            filtered_graph_data.append(graph_entry)

    return filtered_graph_data

# ===================== THE TWO FUNCTIONS BELOW ARE TASKED WITH FILTERING THE DATA AND ASSIGNING LABELS (Regress) ===================

def filter_and_associate_data_question_regress(graph_data, df, question, max_std_dev=1.0):

    filtered_df = df[df[f'{question}_std'] <= max_std_dev]
    filtered_graph_data = []

    # find corresponding row in the filtered DataFrame
    for graph_entry in graph_data:
        meeting = graph_entry['meeting']
        start = graph_entry['start']

        # Find the matching row
        match = filtered_df[(filtered_df['Meeting'] == meeting) & (filtered_df['Start'] * 60 == start)]
        if not match.empty:
            average_score = match[f'{question}_mean'].values[0]
            graph_entry['score'] = average_score
            filtered_graph_data.append(graph_entry)

    return filtered_graph_data


def filter_and_associate_data_category_regress(graph_data, df_kappa, category, min_kappa=0.2):

    kappa_column = f'{category}_Kappa'
    average_column = f'{category}_Average'

    # Filter based on the minimum Kappa score
    filtered_df = df_kappa[df_kappa[kappa_column] >= min_kappa]
    filtered_graph_data = []

    for graph_entry in graph_data:
        meeting = graph_entry['meeting']
        start = graph_entry['start']

        match = filtered_df[(filtered_df['Meeting'] == meeting) & (filtered_df['Start'] * 60 == start)]
        
        if not match.empty:
            average_score = match[average_column].values[0]
            
            graph_entry['score'] = average_score
            filtered_graph_data.append(graph_entry)

    return filtered_graph_data



def filter_and_associate_data_category_binary(graph_data, df_kappa, category, min_kappa=0.2, floor=3.5, ceiling=4.5):

    kappa_column = f'{category}_Kappa'
    average_column = f'{category}_Average'

    filtered_df = df_kappa[(df_kappa[kappa_column] >= min_kappa) & 
                        ((df_kappa[average_column] <= ceiling) | (df_kappa[average_column] >= floor))]

    filtered_graph_data = []

    for graph_entry in graph_data:
        meeting = graph_entry['meeting']
        start = graph_entry['start']

        match = filtered_df[(filtered_df['Meeting'] == meeting) & (filtered_df['Start'] * 60 == start)]
        
        if not match.empty:
            # Binary classification: 1 if average score is above the floor, 0 if below the ceiling
            average_score = match[average_column].values[0]
            graph_entry['score'] = 1 if average_score >= floor else 0
            filtered_graph_data.append(graph_entry)

    return filtered_graph_data

    
# ================ HELPER FUNTION TO REMOVE NAN ENTRIES =================================

def filter_out_nan_entries(dataset):
    filtered_data_list = []
    
    for data in dataset:
        if (torch.isnan(data.x).any() or
            torch.isnan(data.edge_attr).any() or
            torch.isnan(data.y).any()):
            continue
        else:
            filtered_data_list.append(data)

    return filtered_data_list

# get plot object 
def plot_training_data(train_val_targets, suite):
    # Clear the previous figure
    plt.clf()  # Clear the current figure
    plt.close()  # Close any previous figure

    plt.figure(figsize=(5, 4))
    plt.hist(train_val_targets, bins=20, edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {suite} Data')
    plt.xlabel('Target Value (y)')
    plt.ylabel('Frequency')
    plt.grid(True)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return img_base64


def get_filtered_data(read_in_from_user):
    filtered_data = None
    if read_in_from_user['question'] and (read_in_from_user['model_type'] == 'binary'): 
        filtered_data = filter_and_associate_data_question_binary(graphs_cohesion_data, question_level_df, read_in_from_user['question'], 
                                                                    max_std_dev=read_in_from_user['data_parameters']['max_std'], 
                                                                    floor=read_in_from_user['data_parameters']['floor'], 
                                                                    ceiling=read_in_from_user['data_parameters']['ceiling'] )
    elif read_in_from_user['category'] and (read_in_from_user['model_type'] == 'binary'):
        filtered_data = filter_and_associate_data_category_binary(graphs_cohesion_data, category_level_df,get_column(read_in_from_user['category']),
                                                                    min_kappa=read_in_from_user['data_parameters']['min_kappa'], 
                                                                    floor=read_in_from_user['data_parameters']['floor'], 
                                                                    ceiling=read_in_from_user['data_parameters']['ceiling'] )
    elif read_in_from_user['question'] and (read_in_from_user['model_type'] == 'regress'):
        filtered_data = filter_and_associate_data_question_regress(graphs_cohesion_data, question_level_df, read_in_from_user['question'],
                                                                    max_std_dev=read_in_from_user['data_parameters']['max_std'])
    elif read_in_from_user['category'] and (read_in_from_user['model_type'] == 'regress'):
        filtered_data = filter_and_associate_data_category_regress(graphs_cohesion_data, category_level_df, get_column(read_in_from_user['category']),
                                                                    min_kappa=read_in_from_user['data_parameters']['min_kappa'])
    else: 
        print("ERROR: BAD USER INPUT> UNABLE TO RECOGNIZE TYPE OF MODEL")
    
    return filtered_data




# =================== BELOW WE INITIALIZE THE DATA SET IN REFRENCE TO USER INPUT ================

def prepare_dataset(read_in_from_user):
    
    # filter data and create data set 
    filtered_data = get_filtered_data(read_in_from_user)
    dataset = CohesionGraphDataset(filtered_data)

    # split data 
    total_size = len(dataset)
    test_size = int(0.2 * total_size)  
    train_val_size = total_size - test_size 
    data_train_val, data_test = random_split(dataset, [train_val_size, test_size])


    # filter out NaN entries 
    data_train_val = filter_out_nan_entries(data_train_val)
    data_test = filter_out_nan_entries(data_test)

    # get labels 
    train_val_targets = [data.y.item() for data in data_train_val]
    test_targets = [data.y.item() for data in data_test]

    best_alpha = find_optimal_alpha(train_val_targets, data_train_val,  num_bins=2)


    # Lets add a weight to each observation in the training data to correct for imbalances
    # dw = DenseWeight(alpha = find_optimal_alpha(train_val_targets, data_train_val,  num_bins=2))
    # train_val_weights = dw.fit(train_val_targets)
    # for i, data in enumerate(data_train_val):
    #     data.weight = torch.tensor([train_val_weights[i]], dtype=torch.float)
        # print(f'Value: {data.y.item()} | Weight: {data.weight}')

    # Lets obtain a distribution plot for both suites 
    img_base64_train = plot_training_data(train_val_targets, 'Training')
    img_base64_test = plot_training_data(test_targets, 'Testing')

    
    return data_train_val, data_test, img_base64_train, img_base64_test, best_alpha




# ================================== Question prompt ============
questions_dict = {
    # Task Cohesion
    'T1': "Does the team seem to share the responsibility for the task?",
    'T2': "Do you feel that team members share the same purpose/goal/intentions?",
    'T3': "Overall, how enthusiastic is the group?",
    'T4': "How is the morale of the team?",
    'T5': "Overall, do the members give each other a lot of feedback?",
    'T6': "Overall, do the team members appear to be collaborative?",
    'T7': "Does every team member seem to have sufficient time to make their contribution?",

    # Social Cohesion
    'S1': "Overall, do you feel that the work group operates spontaneously?",
    'S2': "Overall, how involved/engaged in the discussion do the participants seem?",
    'S3': "Do the team members seem to enjoy each other's company?",
    'S4': "Does the team seem to have a good rapport?",
    'S5': "Overall, does the atmosphere of the group seem more jovial or serious?",
    'S6': "Overall, does the work group appear to be in tune/in sync with each other?",
    'S7': "Overall, does there appear to be equal participation from the group?",
    'S8': "Overall, do the group members listen attentively to each other?",
    'S9': "Overall, does the team appear to be integrated?",
    'S10': "Do the team members appear to be receptive to each other?",
    'S11': "Do the participants appear comfortable or uncomfortable with each other?",
    'S12': "Is there a strong sense of belonging in the work group?",
    'S13': "Overall, does the atmosphere seem tense or relaxed?",
    'S14': "Does the work group appear to have a strong bond?",
    'S15': "How is the pace of the conversation?",
    'S16': "Overall do the team members seem to be supportive towards each other?",
    'S17': "How well do you think the participants know each other?",

    # Miscellaneous
    'M1': "Is there a leader in the group?",
    'M2': "If you answered YES, does the leader bring the rest of the group together?",
    'M3': "Overall, how cohesive does the group appear?"
}

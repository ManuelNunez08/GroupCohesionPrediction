
#====================================  BELOW ARE ALL THE MODEL PERFORMANCE FUNCTIONS FOR ERC ===============================


#====================================================== GET CONFUSION MATRIX AND STATS ====================================


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
import torch
import numpy as np

def confusion_matrix_plus_stats(all_labels, all_preds, classes):
    
    # Generate confusion matrix and normalize by row
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, values_format=".2f")
    plt.title("Normalized Confusion Matrix (Proportional Accuracy)")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    # Calculate per-class F1 and accuracy scores
    per_class_f1 = f1_score(all_labels, all_preds, average=None, labels=range(len(classes)))
    per_class_accuracy = cm.diagonal()  # Row-normalized matrix diagonal gives per-class accuracy

    # Print per-class metrics
    print("\nPer-Class Accuracy and F1 Scores:")
    for i, class_name in enumerate(classes):
        print(f"{class_name}: Accuracy: {per_class_accuracy[i]:.2f}, F1 Score: {per_class_f1[i]:.2f}")

    # Calculate and print weighted metrics
    weighted_f1_score = f1_score(all_labels, all_preds, average='weighted')
    weighted_accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nWeighted Accuracy (Overall): {weighted_accuracy:.4f}")
    print(f"Weighted F1 Score (Overall): {weighted_f1_score:.4f}")
    
    
    
#====================================================== EVALUATE USING ARGMAX ====================================
    
def get_model_probs_preds_labels(model, loader, device):
    """
    Evaluates the model on the test set, displaying a normalized confusion matrix, per-class accuracy and F1 score, 
    and weighted accuracy and F1 score across all classes.

    :param model: Trained model
    :param classes: List of class names corresponding to labels
    :param test_loader: DataLoader for the test set
    """
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)  # Forward pass
            probs = torch.softmax(output, dim=2).squeeze(1).cpu().numpy()
            predicted_classes = torch.argmax(output, dim=2).squeeze(1)  # Get predicted classes
            all_preds.append(predicted_classes.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
            all_probs.append(probs)

    # Flatten the lists to single arrays
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    return all_probs, all_preds, all_labels
    
    



#================================================GET PROBABILITY DISTRIBUTIONS PER CLASS=================================

    
from collections import defaultdict

def plot_probability_distributions(all_probs, all_preds, all_labels):
    
    # Dictionaries to store probabilities for correct and incorrect predictions
    correct_probs = defaultdict(list)
    incorrect_probs = defaultdict(list)
    
    for probs, pred, label in zip(all_probs, all_preds, all_labels):
        if pred == label:
            correct_probs[label].append(probs)
        else:
            incorrect_probs[label].append(probs)
    
    # Plot histograms for each class using class names
    for class_index, class_name in enumerate(class_names):
        plt.figure(figsize=(12, 6))
        
        # Plot for correct predictions
        if class_index in correct_probs:
            avg_correct_probs = np.mean(correct_probs[class_index], axis=0)
            plt.subplot(1, 2, 1)
            plt.bar(class_names, avg_correct_probs)
            plt.title(f'{class_name}: Avg Probability (Correct)')
            plt.xlabel('Classes')
            plt.ylabel('Probability')
        
        # Plot for incorrect predictions
        if class_index in incorrect_probs:
            avg_incorrect_probs = np.mean(incorrect_probs[class_index], axis=0)
            plt.subplot(1, 2, 2)
            plt.bar(class_names, avg_incorrect_probs)
            plt.title(f'{class_name}: Avg Probability (Incorrect)')
            plt.xlabel('Classes')
            plt.ylabel('Probability')
        
        plt.tight_layout()
        plt.show()
        
        
        

        
        
        
# #====================================================== EVALUATE PREDICTIONS TEMPROALY ====================================
    
# def split_conversations(preds, labels, turns):
#         conversations_preds = []
#         conversations_labels = []
#         conversations_turns = []
        
#         current_preds = []
#         current_labels = []
#         current_turns = []

#         for i in range(len(turns)):
#             if turns[i] == 0 and i != 0:  # New conversation starts when turns[i] == 0
#                 conversations_preds.append(current_preds)
#                 conversations_labels.append(current_labels)
#                 conversations_turns.append(current_turns)
                
#                 current_preds = []
#                 current_labels = []
#                 current_turns = []
            
#             current_preds.append(preds[i])
#             current_labels.append(labels[i])
#             current_turns.append(turns[i])

#         # Append the last conversation
#         if current_preds:
#             conversations_preds.append(current_preds)
#             conversations_labels.append(current_labels)
#             conversations_turns.append(current_turns)
        
#         return conversations_preds, conversations_labels, conversations_turns
        
        
# def get_turn_category(num_turns, bins):
#     """Determine which category a conversation's number of turns belongs to."""
#     if num_turns == bins['single']:
#         return 'single'
#     elif num_turns == bins['double']:
#         return 'double'
#     elif bins['short'][0] <= num_turns <= bins['short'][1]:
#         return 'short'
#     elif bins['mid'][0] <= num_turns <= bins['mid'][1]:
#         return 'mid'
#     elif num_turns >= bins['long']:
#         return 'long'
#     return None
    
    
    
    
    
    
    
    
    
    
    
    
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# def evaluate_model_by_exchange_length(model, classes, test_loader, bins={"single": 1, "double": 2, "short": (3, 10), "mid": (10, 20), "long": 20}):

#     model.eval()  # Set model to evaluation mode
#     model = model.to(device)
#     all_preds = []
#     all_labels = []
#     all_turns = []
    
#     with torch.no_grad():
#         for batch in test_loader:
#             batch = batch.to(device)
#             output = model(batch)  # Forward pass
#             predicted_classes = torch.argmax(output, dim=2).squeeze(1)  # Shape: (batch_size,)
#             all_preds.append(predicted_classes.cpu().numpy())
#             all_labels.append(batch.y.cpu().numpy())
#             all_turns.append(batch.turns.cpu().numpy())

#     # Flatten the lists to single arrays
#     all_preds = np.concatenate(all_preds)
#     all_labels = np.concatenate(all_labels)
#     all_turns = np.concatenate(all_turns)
    
#     # Step 1: Split by conversations 
#     conversations_preds, conversations_labels, conversations_turns = split_conversations(all_preds, all_labels, all_turns)

#     # Dictionary to hold correct and total predictions for each category and percentile
#     categories = ['single', 'double', 'short', 'mid', 'long']
#     correct_by_cat_emotion = {cat: {p: {c: 0 for c in range(len(classes))} for p in range(3)} for cat in categories}
#     total_by_cat_emotion = {cat: {p: {c: 0 for c in range(len(classes))} for p in range(3)} for cat in categories}

#     #Process conversations and fill the dictionaries
#     for preds, labels, turns in zip(conversations_preds, conversations_labels, conversations_turns):
#         num_turns = len(turns)
#         category = get_turn_category(num_turns, bins)

#         if category:
#             for i in range(num_turns):
#                 true_emotion = labels[i]
#                 predicted_emotion = preds[i]
#                 # Assign percentile 
#                 turn_percentile = min(2, i // (num_turns // 3) if num_turns > 2 else i)
#                 # Update total occurrences for the true emotion at this turn
#                 total_by_cat_emotion[category][turn_percentile][true_emotion] += 1
#                 # Update correct predictions if the prediction is correct
#                 if predicted_emotion == true_emotion:
#                     correct_by_cat_emotion[category][turn_percentile][true_emotion] += 1

    
#     # Generate heatmaps for each conversation category
#     for category in categories:
#         # Prepare the numeric accuracy matrix 
#         accuracy_matrix = np.zeros((len(classes), 3)) 
#         fraction_matrix = np.full((len(classes), 3), '', dtype=object)  # Placeholder for fractions

#         for p in range(3):
#             for emotion in range(len(classes)):
#                 if total_by_cat_emotion[category][p][emotion] > 0:
#                     accuracy_matrix[emotion][p] = correct_by_cat_emotion[category][p][emotion] / total_by_cat_emotion[category][p][emotion]
#                     fraction_matrix[emotion][p] = f'{correct_by_cat_emotion[category][p][emotion]} / {total_by_cat_emotion[category][p][emotion]}'
#                 else:
#                     fraction_matrix[emotion][p] = '0 / 0'


#         # Generate the heatmap for shading
#         plt.figure(figsize=(8, 6))
#         ax = sns.heatmap(accuracy_matrix, annot=False, cmap='Blues', xticklabels=['1st Third', '2nd Third', 'Last Third'], yticklabels=classes)

#         # Overlay text (fractions) on top of the heatmap
#         for i in range(len(classes)):
#             for j in range(3):
#                 ax.text(j + 0.5, i + 0.5, fraction_matrix[i][j], color='black', ha='center', va='center', fontsize=12)

#         plt.title(f"Emotion Prediction Accuracy by Percentile - {category.capitalize()} Exchanges")
#         plt.xlabel("Conversation Stage")
#         plt.ylabel("Emotions")
#         plt.show()

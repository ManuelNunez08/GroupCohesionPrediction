from flask import Flask, render_template, request, jsonify, session
from plotly import utils
from json import dumps
from plotting_functions import *
from model_scripts import Binary_GNN, Regress_GNN, CohesionGraphDataset
from model_scripts.CohesionGraphDataset import question_level_df, category_level_df, questions_dict

# ======================= APP SETUP =====================================
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# ====================== HELPER FUNCTIONS =====================================

def initialize_session():
    """Initialize the session with default values."""
    if 'model_metadata' not in session:
        session['model_metadata'] = {
            'model_type': None,
            'question': None,
            'question_prompt': None,
            'category': None,
            'data_parameters': {
                'floor': None,
                'ceiling': None,
                'min_kappa': None,
                'max_std': None,
            },
            'hyperparameters': {
                'batch_size': None,
                'save_loss_interval': None,
                'print_interval': None,
                'n_epochs': None,
                'learning_rate': None,
                'alpha_weight': None,
            }
        }


def get_plot():
    plot_data, layout, config = None, None, None
    model_metadata = session['model_metadata']

    if model_metadata['question']:
        if model_metadata['model_type'] == 'regress':
            plot_data, layout, config = plot_std_vs_mean_regress(question_level_df, model_metadata['question'],
                                                        floor=model_metadata['data_parameters']['floor'],
                                                        ceiling=model_metadata['data_parameters']['ceiling'],
                                                        max_std=model_metadata['data_parameters']['max_std'])
        else:
            plot_data, layout, config = plot_std_vs_mean_binary(question_level_df, model_metadata['question'],
                                                        floor=model_metadata['data_parameters']['floor'],
                                                        ceiling=model_metadata['data_parameters']['ceiling'],
                                                        max_std=model_metadata['data_parameters']['max_std'])
    elif model_metadata['category']:
        if model_metadata['model_type'] == 'regress':
            plot_data, layout, config = plot_kappa_vs_average_regress(category_level_df, model_metadata['category'],
                                                            floor=model_metadata['data_parameters']['floor'],
                                                            ceiling=model_metadata['data_parameters']['ceiling'],
                                                            kappa_threshold=model_metadata['data_parameters']['min_kappa'])
        else:
            plot_data, layout, config = plot_kappa_vs_average_binary(category_level_df, model_metadata['category'],
                                                            floor=model_metadata['data_parameters']['floor'],
                                                            ceiling=model_metadata['data_parameters']['ceiling'],
                                                            kappa_threshold=model_metadata['data_parameters']['min_kappa'])

    return plot_data, layout, config


def set_keys_to_none(d):
    for key, value in d.items():
        if isinstance(value, dict):  
            set_keys_to_none(value)  
        else:
            d[key] = None 


# ======================= BELOW IS THE APP ======================================

@app.before_request
def before_request():
    initialize_session()

@app.route('/')
def home():
    # Set all keys to point to None
    session.clear()
    initialize_session()
    print("Session data home:\n ", session['model_metadata'])
    return render_template('home.html')



# ============================ Data Selction ==================================

@app.route('/data_proc', methods=['POST'])
def data_proc():
    
    model_metadata = session['model_metadata']

    model_metadata['question'] = request.form.get('question')
    model_metadata['category'] = request.form.get('category')
    model_metadata['model_type'] = request.form.get('model_type')

    model_metadata['data_parameters']['floor'] = 4.0
    model_metadata['data_parameters']['ceiling'] = 4.0
    model_metadata['data_parameters']['max_std'] =  3.0
    model_metadata['data_parameters']['min_kappa'] = -0.3

    print("Session data after data_proc load:\n ", session['model_metadata'])

    if model_metadata['question']:
        model_metadata['question_prompt'] = questions_dict[model_metadata['question']]
    else:
        model_metadata['question_prompt'] = ''

    print("Session data after data_proc load:\n ", session['model_metadata'])

        



    session['model_metadata'] = model_metadata



    plot_data, layout, config = get_plot()
    data_json = dumps(plot_data, cls=utils.PlotlyJSONEncoder)
    layout_json = dumps(layout, cls=utils.PlotlyJSONEncoder)
    config_json = dumps(config, cls=utils.PlotlyJSONEncoder)

    return render_template('data_proc.html', model_metadata=session['model_metadata'], data_JSON=data_json, layout_JSON=layout_json, config_JSON=config_json)

@app.route('/update_plot', methods=['POST'])
def update_plot():
    model_metadata = session['model_metadata']

    # Update data parameters
    model_metadata['data_parameters']['floor'] = float(request.get_json().get('floor', 4.0))
    model_metadata['data_parameters']['ceiling'] = float(request.get_json().get('ceiling', 4.0))
    model_metadata['data_parameters']['max_std'] = float(request.get_json().get('threshold_std', 2.0))
    model_metadata['data_parameters']['min_kappa'] = float(request.get_json().get('threshold_kappa', -0.2))

    session['model_metadata'] = model_metadata

    print("Session data After slider slide:\n ", session['model_metadata'])


    plot_data, layout, config = get_plot()
    data_json = dumps(plot_data, cls=utils.PlotlyJSONEncoder)
    layout_json = dumps(layout, cls=utils.PlotlyJSONEncoder)
    config_json = dumps(config, cls=utils.PlotlyJSONEncoder)

    return jsonify({'plotData': data_json, 'layout': layout_json, 'config': config_json})





# ===================== INTERACTIVE BUILD ================
@app.route('/interactive_build', methods=['POST', 'GET'])
def interactive_build():
        # Prepare the dataset and get the distribution graph
    data_train_val, data_test, _, _, alpha = CohesionGraphDataset.prepare_dataset(session['model_metadata'])
    alpha = round(alpha, 2)


    # Store the datasets in session
    global curr_data_test
    curr_data_test = data_test
    global curr_data_train_val
    curr_data_train_val = data_train_val

    # Store the datasets in session
    global saved_data_test
    saved_data_test = curr_data_test
    global saved_data_train_val
    saved_data_train_val = curr_data_train_val

    train_val_targets = [data.y.item() for data in data_train_val]
    test_targets = [data.y.item() for data in data_test]

    plot_data_train, layout_train, config_train = plot_suite_data_plotly(train_val_targets, 'Training')
    plot_data_test, layout_test, config_test = plot_suite_data_plotly(test_targets, 'Testing')

    # save as json objects 
    data_train_json = dumps(plot_data_train, cls=utils.PlotlyJSONEncoder)
    layout_train_json = dumps(layout_train, cls=utils.PlotlyJSONEncoder)
    config_train_json = dumps(config_train, cls=utils.PlotlyJSONEncoder)
    data_test_json = dumps(plot_data_test, cls=utils.PlotlyJSONEncoder)
    layout_test_json = dumps(layout_test, cls=utils.PlotlyJSONEncoder)
    config_test_json = dumps(config_test, cls=utils.PlotlyJSONEncoder)


    # Update session data with new parameters
    model_metadata = session['model_metadata']
    model_metadata['hyperparameters']['alpha_weight'] = alpha 
    model_metadata['hyperparameters']['batch_size'] = 4 
    model_metadata['hyperparameters']['n_epochs'] = 250  
    model_metadata['hyperparameters']['learning_rate'] = 0.01 
    model_metadata['hyperparameters']['dropout_rate'] = 0.5  
    model_metadata['hyperparameters']['save_loss_interval'] = 10
    model_metadata['hyperparameters']['print_interval'] = 50
    session['model_metadata'] = model_metadata




    return render_template('interactive_build.html', 
                        data_train_JSON=data_train_json, layout_train_JSON=layout_train_json, config_train_JSON=config_train_json, 
                        data_test_JSON=data_test_json, layout_test_JSON=layout_test_json, config_test_JSON=config_test_json,
                        model_metadata=session['model_metadata'], 
                        alpha=alpha)


@app.route('/update_parameters', methods=['POST', 'GET'])
def update_parameters():
    # Get the JSON data from the AJAX request
    data = request.get_json()

    # Update session data with new parameters
    model_metadata = session['model_metadata']
     # Convert to float for alpha_weight and learning_rate, and integer for batch_size and n_epochs
    model_metadata['hyperparameters']['alpha_weight'] = float(data.get('alpha'))  # Cast to float
    model_metadata['hyperparameters']['batch_size'] = int(data.get('batch_size'))  # Cast to int
    model_metadata['hyperparameters']['n_epochs'] = int(data.get('epochs'))  # Cast to int
    model_metadata['hyperparameters']['learning_rate'] = float(data.get('learning_rate'))  # Cast to float
    model_metadata['hyperparameters']['dropout_rate'] = float(data.get('dropout_rate'))  # Cast to floa
    model_metadata['hyperparameters']['save_loss_interval'] = 10
    model_metadata['hyperparameters']['print_interval'] = 50
    
    # Assign back to session
    session['model_metadata'] = model_metadata

    # Send a success response back to the client
    return jsonify({'message': 'Parameters updated successfully'})



@app.route('/regenerateData', methods=['POST', 'GET'])
def regenerateData():
        # Prepare the dataset and get the distribution graph
    data_train_val, data_test, _, _, alpha = CohesionGraphDataset.prepare_dataset(session['model_metadata'])
    alpha = round(alpha, 2)

    # Store the datasets in session
    global curr_data_test
    curr_data_test = data_test
    global curr_data_train_val
    curr_data_train_val = data_train_val

    train_val_targets = [data.y.item() for data in data_train_val]
    test_targets = [data.y.item() for data in data_test]

    plot_data_train, layout_train, config_train = plot_suite_data_plotly(train_val_targets, 'Training')
    plot_data_test, layout_test, config_test = plot_suite_data_plotly(test_targets, 'Testing')

    # save as json objects 
    data_train_json = dumps(plot_data_train, cls=utils.PlotlyJSONEncoder)
    layout_train_json = dumps(layout_train, cls=utils.PlotlyJSONEncoder)
    config_train_json = dumps(config_train, cls=utils.PlotlyJSONEncoder)
    data_test_json = dumps(plot_data_test, cls=utils.PlotlyJSONEncoder)
    layout_test_json = dumps(layout_test, cls=utils.PlotlyJSONEncoder)
    config_test_json = dumps(config_test, cls=utils.PlotlyJSONEncoder)


    return jsonify({'plotData_train': data_train_json, 'layout_train': layout_train_json, 'config_train': config_train_json,
                    'plotData_test': data_test_json, 'layout_test': layout_test_json, 'config_test': config_test_json, 'alpha': alpha})


@app.route('/saveDataSuites', methods=['POST', 'GET'])
def saveDataSuites():

    # Store the datasets in session
    global saved_data_test
    saved_data_test = curr_data_test
    global saved_data_train_val
    saved_data_train_val = curr_data_train_val


    train_val_targets = [data.y.item() for data in saved_data_train_val]
    test_targets = [data.y.item() for data in saved_data_test]

    plot_data_train, layout_train, config_train = plot_suite_data_plotly(train_val_targets, 'Training')
    plot_data_test, layout_test, config_test = plot_suite_data_plotly(test_targets, 'Testing')

    # save as json objects 
    data_train_json = dumps(plot_data_train, cls=utils.PlotlyJSONEncoder)
    layout_train_json = dumps(layout_train, cls=utils.PlotlyJSONEncoder)
    config_train_json = dumps(config_train, cls=utils.PlotlyJSONEncoder)
    data_test_json = dumps(plot_data_test, cls=utils.PlotlyJSONEncoder)
    layout_test_json = dumps(layout_test, cls=utils.PlotlyJSONEncoder)
    config_test_json = dumps(config_test, cls=utils.PlotlyJSONEncoder)


    return jsonify({'plotData_train': data_train_json, 'layout_train': layout_train_json, 'config_train': config_train_json,
                    'plotData_test': data_test_json, 'layout_test': layout_test_json, 'config_test': config_test_json})



@app.route('/trainAndTestInteractive', methods=['POST', 'GET'])
def trainAndTestInteractive():
    model_metadata = session['model_metadata']
    
    print("Session data just before training: ", model_metadata)
    # Proceed with training
    if model_metadata['model_type'] == 'binary':
        Binary_GNN.train_model(saved_data_train_val, model_metadata['hyperparameters'])
        kmodels = Binary_GNN.get_results(saved_data_test, model_metadata['hyperparameters'])
    elif model_metadata['model_type'] == 'regress':
        Regress_GNN.train_model(saved_data_train_val, model_metadata['hyperparameters'])
        kmodels = Regress_GNN.get_results(saved_data_test, model_metadata['hyperparameters'])
    else:
        print("ERROR: Could not identify the correct model to train")

    
    # model_metadata = dumps(model_metadata, default=str)

    return jsonify(kmodels=kmodels, model_metadata=model_metadata)



if __name__ == '__main__':
    app.run(debug=True)

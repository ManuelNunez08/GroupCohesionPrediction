// Tab switching logic
document.querySelectorAll('.tab-link').forEach(link => {
  link.addEventListener('click', function() {
    const tabId = this.getAttribute('data-tab');
    document.querySelectorAll('.tab-content').forEach(content => {
      content.style.display = 'none';
    });
    document.getElementById(tabId).style.display = 'block';
  });
});






document.addEventListener('DOMContentLoaded', function() {
    
    save_alpha_button = document.getElementById('save-alpha-btn');
    save_hyperparams_button = document.getElementById('save-hyperparams-btn');
    regenerate_data_split = document.getElementById('regenerate-split-btn');
    save_sata_split = document.getElementById('save-split-btn');
    trainAndTest_button = document.getElementById('model-output-btn');
    retrain_button = document.getElementById('model-retrain')

    // Listen to changes in value
    save_alpha_button.addEventListener('click', updateParams);
    save_hyperparams_button.addEventListener('click', updateParams);
    regenerate_data_split.addEventListener('click', regenerateData);
    save_sata_split.addEventListener('click', saveData);
    trainAndTest_button.addEventListener('click', trainAndTest)
    retrain_button.addEventListener('click', resetSelection)

    function resetSelection(event) {
        event.preventDefault()
        // Clear previous results if necessary
        const resultsDiv = document.getElementById('model-results-tabs');
        resultsDiv.innerHTML = '';
        const resultsHeaderDiv = document.getElementById('model-results-header');
        resultsHeaderDiv.innerHTML = '';
        document.getElementById('train-row').style.display = 'block';
        document.getElementById('retrain-row').style.display = 'none';

    }

    function updateParams(event) {

        event.preventDefault()

        const alpha = document.getElementById('alpha').value;
        const batch_size = document.getElementById('batch_size').value;
        const learning_rate = document.getElementById('learning_rate').value;
        const dropout_rate = document.getElementById('dropout_rate').value;
        const epochs = document.getElementById('epochs').value;


        const postData = {
            alpha: alpha,
            batch_size: batch_size, 
            learning_rate: learning_rate, 
            dropout_rate: dropout_rate,
            epochs: epochs  
        };

        fetch('/update_parameters', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(postData),
        })
        .then(response => response.json())
        .then(data => {
            // Update the displayed saved values in the "Saved Parameters and Data" section
            document.getElementById('savedAlpha').textContent = alpha;
            document.getElementById('savedBatchSize').textContent = batch_size;
            document.getElementById('savedEpochs').textContent = epochs;
            document.getElementById('savedLearningRate').textContent = learning_rate;
            document.getElementById('savedDropoutRate').textContent = dropout_rate;
        })
        .catch(error => {
            console.error('Error updating plot:', error);
        });
    }


    function regenerateData(event) {

        event.preventDefault()

        fetch('/regenerateData', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            var updated_data_train = JSON.parse(data.plotData_train);
            var updated_layout_train = JSON.parse(data.layout_train);
            var config_train = JSON.parse(data.config_train);

            var updated_data_test = JSON.parse(data.plotData_test);
            var updated_layout_test = JSON.parse(data.layout_test);
            var config_test = JSON.parse(data.config_test);
            var updated_alpha_rec = data.alpha;

            document.getElementById('Alpha_rec').textContent = updated_alpha_rec;
            console.log('Changed Alpha Rec')

            // Update the graph with new data
            Plotly.react('training_data', updated_data_train, updated_layout_train, config_train);
            Plotly.react('testing_data', updated_data_test, updated_layout_test, config_test);
        })
        .catch(error => {
            console.error('Error updating plot:', error);
        });
    }

    function saveData(event) {

        event.preventDefault()

        fetch('/saveDataSuites', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            var updated_data_train = JSON.parse(data.plotData_train);
            var updated_layout_train = JSON.parse(data.layout_train);
            var config_train = JSON.parse(data.config_train);

            var updated_data_test = JSON.parse(data.plotData_test);
            var updated_layout_test = JSON.parse(data.layout_test);
            var config_test = JSON.parse(data.config_test);

            // Update the graph with new data
            Plotly.react('training_data_main', updated_data_train, updated_layout_train, config_train);
            Plotly.react('testing_data_main', updated_data_test, updated_layout_test, config_test);
        })
        .catch(error => {
            console.error('Error updating plot:', error);
        });
    }


    function trainAndTest(event) {
    event.preventDefault();

    fetch('/trainAndTestInteractive', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        const kmodels = data.kmodels;
        console.log("Data", data);  // Add this line to debug
        const model_metadata = data.model_metadata;

        // Clear previous results if necessary
        const resultsDiv = document.getElementById('model-results-tabs');
        resultsDiv.innerHTML = '';

        const resultsHeaderDiv = document.getElementById('model-results-header');
        resultsHeaderDiv.innerHTML = '';

        // Create tab navigation and content containers
        const navTabs = document.createElement('ul');
        navTabs.classList.add('nav', 'nav-tabs', 'mt-4');

        const tabContentContainer = document.createElement('div');
        tabContentContainer.classList.add('tab-content', 'mt-4');

        document.getElementById('train-row').style.display = 'none';
        document.getElementById('retrain-row').style.display = 'block';


        resultsHeaderDiv.innerHTML = `
                    <div class="row">
                    <div class="col-md-12">
                    <h3>Model Results</h3>
                                <p><strong>${model_metadata['question_prompt'] ? 'Question' : 'Category'}:</strong> 
                                ${model_metadata['question_prompt'] ? model_metadata['question_prompt'] : model_metadata['category']}</p>
                                <p><strong>Model Parameters</strong>:
                                                            Alpha: ${model_metadata['hyperparameters']['alpha_weight']} <strong>| </strong>
                                                            Batch Size: ${model_metadata['hyperparameters']['batch_size']} <strong>| </strong>
                                                            Epochs: ${model_metadata['hyperparameters']['n_epochs']} <strong>| </strong>
                                                            Learning Rate: ${model_metadata['hyperparameters']['learning_rate']} <strong>| </strong>
                                                            Dropout Rate: ${model_metadata['hyperparameters']['dropout_rate']}

                                </p>
                    </div>
                </div>
        `;

        kmodels.forEach((model, index) => {
            const perfDic = model[0];  // Performance dictionary
            const regressPlot = model[1];  // Regression plot (could be a Plotly figure)

            // Create tab buttons
            const navTabItem = document.createElement('li');
            navTabItem.classList.add('nav-item');
            const navTabLink = document.createElement('a');
            navTabLink.classList.add('nav-link');
            navTabLink.href = `#fold-${index}`;
            navTabLink.innerText = `${index + 1}-fold`;
            navTabLink.setAttribute('data-toggle', 'tab');
            navTabItem.appendChild(navTabLink);
            navTabs.appendChild(navTabItem);

            // Create tab content for each fold
            const tabPane = document.createElement('div');
            tabPane.classList.add('tab-pane', 'fade');
            tabPane.id = `fold-${index}`;

            // Activate the first tab by default
            if (index === 0) {
                navTabLink.classList.add('active');
                tabPane.classList.add('active', 'show');
            }

            // Add two-column layout for metrics and plot
            tabPane.innerHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <h5>Performance Metrics</h5>
                        <ul>
                            ${Object.keys(perfDic).map(key => `<li><strong>${key}:</strong> ${perfDic[key]}</li>`).join('')}
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h5>Regression Plot</h5>
                        <img src="data:image/png;base64,${regressPlot}" alt="Test Data Distribution">
                    </div>
                </div>
            `;

            // Append tab content
            tabContentContainer.appendChild(tabPane);
        });

        // Append the navigation tabs and content container to the results div
        resultsDiv.appendChild(navTabs);
        resultsDiv.appendChild(tabContentContainer);
    })
    .catch(error => {
            console.error('Error:', error);
        });
    }

});
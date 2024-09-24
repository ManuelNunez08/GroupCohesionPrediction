// ==================================== DATA PROC PAGE =============================================

// Define slider variables globally
let floorSlider;
let ceilingSlider;
let kappaSlider;
let stdSlider;



// Dynamically customizes Intro header and sliders shown 
document.addEventListener('DOMContentLoaded', function() {
    const question = modelMetadata.question;
    const category = modelMetadata.category;
    const selectedModel = modelMetadata.model_type;

    if (question && !category && selectedModel === 'binary') {
        document.getElementById('BinaryQuestionHeader').style.display = 'block';
        document.getElementById('stdDevDiv').style.display = 'block';
        document.getElementById('kappaDiv').style.display = 'none';
    } else if (question && !category && selectedModel === 'regress') {
        document.getElementById('RegressQuestionHeader').style.display = 'block';
        document.getElementById('ceilingSlider').style.display = 'none';
        document.getElementById('floorSlider').style.display = 'none';
        document.getElementById('stdDevDiv').style.display = 'block';
        document.getElementById('kappaDiv').style.display = 'none';
    } else if (!question && category && selectedModel === 'regress') {
        document.getElementById('RegressCategoryHeader').style.display = 'block';
        document.getElementById('ceilingSlider').style.display = 'none';
        document.getElementById('floorSlider').style.display = 'none';
        document.getElementById('kappaDiv').style.display = 'block';
        document.getElementById('stdDevDiv').style.display = 'none';
    } else if (!question && category && selectedModel === 'binary') {
        document.getElementById('BinaryCategoryHeader').style.display = 'block';
        document.getElementById('kappaDiv').style.display = 'block';
        document.getElementById('stdDevDiv').style.display = 'none';
    }

    // Initialize slider variables
    floorSlider = document.getElementById('floor');
    ceilingSlider = document.getElementById('ceiling');
    kappaSlider = document.getElementById('kappa');
    stdSlider = document.getElementById('std_dev');

    // Listen to changes in the sliders
    floorSlider.addEventListener('input', updatePlot);
    ceilingSlider.addEventListener('input', updatePlot);
    kappaSlider.addEventListener('input', updatePlot);
    stdSlider.addEventListener('input', updatePlot);

    const floorValDisplay = document.getElementById('floorVal');
    const ceilingValDisplay = document.getElementById('ceilingVal');

    function updatePlot() {
        const floor = floorSlider.value;
        const ceiling = ceilingSlider.value;
        const thresholdStd = stdSlider.value;
        const thresholdKappa = kappaSlider.value;

        const postData = {
            floor: floor,
            ceiling: ceiling,
            threshold_std: thresholdStd,
            threshold_kappa: thresholdKappa,
        };

        fetch('/update_plot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(postData),
        })
        .then(response => response.json())
        .then(data => {
            var updated_data = JSON.parse(data.plotData);
            var updated_layout = JSON.parse(data.layout);
            var config = JSON.parse(data.config);

            console.log('plot Data:', updated_data);
            console.log('Layout', updated_layout);
            console.log('Config:', config);

            // Update the graph with new data
            Plotly.react('annotators_data', updated_data, updated_layout, config);
        })
        .catch(error => {
            console.error('Error updating plot:', error);
        });
    }

    // Event listeners for updating plot in real-time in reference to sliders 
    floorSlider.addEventListener('input', function() {
        let floorValue = parseFloat(floorSlider.value);
        let ceilingValue = parseFloat(ceilingSlider.value);

        floorValDisplay.innerHTML = floorValue;

        // Make sure the ceiling value is <= floor value
        if (ceilingValue > floorValue) {
            ceilingSlider.value = floorValue;
            ceilingValDisplay.innerHTML = floorValue;
        }
        
        updatePlot();
    });
    ceilingSlider.addEventListener('input', function() {
        let ceilingValue = parseFloat(ceilingSlider.value);
        let floorValue = parseFloat(floorSlider.value);
        
        ceilingValDisplay.innerHTML = ceilingValue;

        if (ceilingValue > floorValue) {
            floorSlider.value = ceilingValue;
            floorValDisplay.innerHTML = ceilingValue;
        }

        updatePlot();
    });
});


// ==================================== HOME PAGE =============================================
let selectedQuestion = null;
let selectedCategory = null;
let selectedModel = null;

document.addEventListener("DOMContentLoaded", function() {


    continueButton = document.getElementById('continue-btn')
    continueButton.addEventListener('click', continueToNext);


    // Handle selection of a question from the dropdown
    const questionSelect = document.getElementById('questionSelect');
    questionSelect.addEventListener('change', handleChange);

    function handleChange(){
        const selectedOption = this.options[this.selectedIndex];
        const questionValue = selectedOption.value;

        if (questionValue.startsWith('S') || questionValue.startsWith('T') || questionValue.startsWith('M')) {
            selectQuestion(questionValue, selectedOption);
        } else {
            selectCategory(questionValue, selectedOption);
        }
    }


    // Handle selection of a model from the dropdown
    const modelSelect = document.getElementById('modelTypeSelect');
    modelSelect.addEventListener('change', function() {
        const selectedOption = this.options[this.selectedIndex];
        const modelValue = selectedOption.value;
        selectModelType(modelValue);
    });



    // Reset the style for the previously selected question and category (home page)
    function resetSelection() {
        selectedQuestion = null;
        selectedCategory = null;
    }

    // select question 
    function selectQuestion(question, element) {
        resetSelection();  // Reset category selection if a question is selected
        selectedQuestion = question;
        console.log("Selected Question:", selectedQuestion);
    }

    // select category 
    function selectCategory(category, element) {
        resetSelection();  // Reset question selection if a category is selected
        selectedCategory = category;
        console.log("Selected Category:", selectedCategory);
    }

    // Highlight selected model and store the selection
    function selectModelType(modelType) {
        selectedModel = modelType;
        console.log("Selected Model Type:", selectedModel);
    }

    // Redirect to the data processing page with data 
    function continueToNext() {
        if (!selectedModel || (!selectedCategory && !selectedQuestion)) {
            alert("Please select a question or category and a model type before continuing.");
            return;
        }

        document.getElementById('selectedQuestion').value = selectedQuestion;
        document.getElementById('selectedCategory').value = selectedCategory;
        document.getElementById('selectedModel').value = selectedModel; 

        // Dynamically set the form action based on the model type
        const form = document.getElementById('continueForm');
        form.action = "/data_proc";  
        form.submit();

        // Clear the selections after submission
        resetSelection();
    }



});



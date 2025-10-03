document.getElementById('exoplanet-form').onsubmit = async function(e) {
    e.preventDefault();
    const form = e.target;
    const data = {};
    
    for (let el of form.elements) {
        if (el.name) {
            data[el.name] = el.value === "" ? null : Number(el.value);
        }
    }
    
    const resultElem = document.getElementById('result');
    resultElem.className = 'loading';
    resultElem.innerText = "Analyzing data...";
    
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({user_input: data})
        });
        
        if (!response.ok) {
            throw new Error("Prediction failed");
        }
        
        const result = await response.json();
        resultElem.className = '';
        
        if (result.predicted_label) {
            resultElem.innerText = "Predicted Class: " + result.predicted_label + " (Code: " + result.predicted_class + ")";
        } else {
            resultElem.innerText = "Predicted Class Code: " + result.predicted_class;
        }
    } catch (err) {
        resultElem.className = '';
        resultElem.innerText = "Error: " + err.message;
    }
}
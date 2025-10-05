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
        const response = await fetch('http://localhost:5000/predict_tess', {
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

async function uploadFile() {
  const fileInput = document.getElementById('file-input');
  const file = fileInput.files[0];
  if (!file) {
    alert("Please select a file first.");
    return;
  }

  const resultElem = document.getElementById('result');
  resultElem.className = 'loading';
  resultElem.innerText = "Uploading & analyzing file...";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("http://localhost:5000/upload_tess", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      throw new Error("File prediction failed");
    }

    const result = await response.json();
    resultElem.className = '';
    // Handle array of predictions
    if (Array.isArray(result)) {
      let output = "Predictions:<br><ul>";
      result.forEach((item, idx) => {
        output += `<li>Row ${idx + 1}: ${item.predicted_label}</li>`;
      });
      output += "</ul>";
      resultElem.innerHTML = output;
    } else {
      // Fallback for single prediction
      resultElem.innerText = "Predicted Class: " + result.predicted_label ;
    }
  } catch (err) {
    resultElem.className = '';
    resultElem.innerText = "Error: " + err.message;
  }
}

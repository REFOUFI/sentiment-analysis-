//========================================================================
// Drag and drop image handling
//========================================================================

var fileDrag = document.getElementById("file-drag");
var fileSelect = document.getElementById("file-upload");

// Add event listeners
fileDrag.addEventListener("dragover", fileDragHover, false);
fileDrag.addEventListener("dragleave", fileDragHover, false);
fileDrag.addEventListener("drop", fileSelectHandler, false);
fileSelect.addEventListener("change", fileSelectHandler, false);

function fileDragHover(e) {
  // prevent default behaviour
  e.preventDefault();
  e.stopPropagation();

  fileDrag.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
}

function fileSelectHandler(e) {
  // handle file selecting
  var files = e.target.files || e.dataTransfer.files;
  fileDragHover(e);
  for (var i = 0, f; (f = files[i]); i++) {
    previewFile(f);
  }
}

//========================================================================
// Main button events
//========================================================================

function submitText() {
  // action for the submit button
  console.log("submit");
  var textBox = document.getElementById("text_input");

  var predResult = document.getElementById("pred-result");
  predResult.innerHTML = "";

  if (textBox.value.trim().length == 0) {
    window.alert("Please enter text");
    return;
  }
  var loader = document.getElementById("loader");
  show(loader);

  // call the predict function of the backend
  predictText(textBox.value);
}

function clearText() {

  // remove text
  var textBox = document.getElementById("text_input");
  var predResult = document.getElementById("pred-result");
  textBox.value = "";
  predResult.innerHTML = "";
}


//========================================================================
// Helper functions
//========================================================================

function predictText(Text) {
  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({"text" : Text})
  })
    .then(resp => {
      if (resp.ok)
        resp.json().then(data => {
          displayResult(data);
        });
    })
    .catch(err => {
        var loader = document.getElementById("loader");
        hide(loader);
      console.log("An error occured", err.message);
      window.alert("Oops! Something went wrong.");
    });
}


function displayResult(data) {
  var loader = document.getElementById("loader");
  hide(loader);
  var predResult = document.getElementById("pred-result");

  predResult.innerHTML = capitalizeFLetter(data.result);
}

function hide(el) {
  // hide an element
  el.classList.add("hidden");
}

function show(el) {
  // show an element
  el.classList.remove("hidden");
}

function capitalizeFLetter(string) {

    return string[0].toUpperCase() + string.slice(1);
  }

window.onload = main;

function main(){

    const inputFile = document.getElementById('inputFile')
    const preview = document.getElementById('preview');
    const form = document.querySelector('form');
    const submit = document.getElementById('submit');
    const result = document.getElementById('result');

    inputFile.addEventListener('change',function(evt){
        const file = evt.target.files[0];
        if(!file){
            return;
        }
        const fr = new FileReader();
        fr.onload = function(e){
            const image = document.createElement('img');
            image.src=e.target.result;
            preview.innerHTML = '';
            preview.append(image);
        };
        fr.readAsDataURL(file);
    });

    form.addEventListener('submit',function(evt){
        evt.preventDefault();
        afficherResultat();
       /*fetch('upload', {
            method: 'POST',
            body: formData
            })
            .then(function(response){
                return response.text();
            })
            .then(function(message){
                result.textContent = message;
                result.className = 'show success';
            })
            .catch(function(error){
                result.textContent="Le fichier n'a pas pu être uploader...";
                result.className = 'show error';
            });*/   
    });

    async function afficherResultat(){
        formData = new FormData(form); 
        try{
            const reponse = await fetch('upload', {
                method: 'POST',
                body: formData
            });
            
            const message = await reponse.text();
            
            if (!reponse.ok) {
                result.textContent = message;
                result.className = 'show error';
            } else {
                result.textContent = message;
                result.className = 'show success';
            }
            
        } catch(error) {
            result.textContent = "Le fichier n'a pas pu être uploadé...";
            result.className = 'show error';
        }
    }

}
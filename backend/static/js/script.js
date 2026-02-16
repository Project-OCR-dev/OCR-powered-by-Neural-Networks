window.onload = main;

function main(){

    const inputFile = document.getElementById('inputFile')
    const preview = document.getElementById('preview');

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
}
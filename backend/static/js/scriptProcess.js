window.onload = function(){
    const path= window.location.pathname;
    const filename= path.split('/')[2];
    console.log(filename);

    setTimeout(async function(){
        const url = `/analyze/${filename}`;
        try{
            const reponse = await fetch(url, {
                method: 'POST'
            });
            window.location.href = reponse.url;
            
        }catch(error){
            console.error(error);
        }
        
    }
    ,4000);

}
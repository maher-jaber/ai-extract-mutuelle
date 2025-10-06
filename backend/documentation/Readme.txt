Cet api permet de extraire des informations depuis des mutuelle de pays France .
L'api utilise Flask comme moteur api .
l'api presente 3 endpoint principaux : 
- [GET] /api/health => verifier l'etat de api : down ou live 
- [POST] /api/login => pour s'authentifier :
* username: admin
* password : password123
dans cet endpoint on a utilisé le jwt token et refresh token , donc si authententification est bonne tu aura les deux 
biensure le refresh token tu le sauvegarde au front pour le renvoyer si tu as status 401 pour avoir un nouveau token
Tu peux renvoyer le refresh token au endpoint [POST] /api/refresh (Authorization: Bearer <refresh_token>)

- [POST] /api/process => endpoint principal : il accepte un fichier pdf en formdata et renvoie un json bien structuré comme montre l'exemple en piece jointe 

tu trouvera la documentation swagger http://192.168.24.94/apidocs

Deux mutuelles pour tester en piece jointe , tu peut tester via l'interface : http://192.168.24.94/

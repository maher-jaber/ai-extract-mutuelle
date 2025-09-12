# Étape 1 : utiliser l’image officielle Nginx
FROM nginx:alpine

# Étape 2 : copier ton fichier HTML dans le dossier par défaut de Nginx
COPY index.html /usr/share/nginx/html/index.html

# Étape 3 : exposer le port
EXPOSE 80

# Étape 4 : démarrer Nginx
CMD ["nginx", "-g", "daemon off;"]

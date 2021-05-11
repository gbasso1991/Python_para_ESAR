# Python_para_ESAR
Repositorio para los programas en Matlab y Python

Para gestionar el reposotorio puede usarse Visual Studio Code o bien instalando GIT (https://git-scm.com/downloads).
Para usar este último desde el correspondiente prompt, dejo una cheatsheet:


- Creamos el repositorio remoto.

- Clonamos el repositorio:
 ```
-git clone *url del repo*
```

- Nos movemos a una nueva rama:
```
 -git checkout -b *nombre de nueva rama*
```

- Hacemos los cambios sobre dicha rama:
```
  -git status (opcional)
  -git add . (o nombre del archivo a agregar)
  -git commit -m *mensaje del commit*
 ```

- Enviamos nuestra rama local al repositorio remoto:
```
  -git push origin *nombre de la rama*
```  

- En github o el host que usemos para nuestro repositorio creamos una PULL REQUEST.

- Una vez aprobados los cambios mergeamos nuestra PR. (Los cambios de nuestra rama ya pasarian a estar en la rama master o la rama destino que corresponda)

- Actualizamos nuestro master local para obtener los nuevos cambios:
```  
-git pull origin master
```

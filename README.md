# AI Educational Materials


<p align="center">
  <a href="#how-to-contribute">How to Contribute</a> •
  <a href="#license">License</a>
</p>
<p align="center" markdown="1">
    <a href="https://github.com/sut-ai/notes/actions/workflows/main.yml" >
        <img src="https://github.com/sut-ai/notes/actions/workflows/main.yml/badge.svg?branch=master" alt="Webify & Deploy">
    </a>
</p>

This repository contains Lecture Notes regarding artificial intelligence course. 

## How to Contribute
Right now we only accept educational notebooks, if you want to submit your notebook, do as follows: 
1. First [fork this repository](https://github.com/sut-ai/notes/fork)!
2. Create a new folder in `notebooks/` (please follow lower-cased underline seperated naming convention for your folder).
3. Create your notebook inside your newly created folder and name it `index.md`. Make sure that all the necessary files (images and other potential assets) are all located inside this folder. Make sure that all your assets are referenced relative from your notebook.
4. Create `matadata.yml` inside your notebook and fill in these informations:

    ```yml
    title: <change this> # shown on browser tab
    
    header:
        title: <change this> # title of your notebook
        description: <change this> # short description of your notebook
    
    authors:
        label: 
            position: top
        kind: people
        content:
        # list of notebook authors
        - name: <change this> # name of author
          role: Author # change this if you want
          contact:
          # list of contact information
          - link: https://github.com/<your_github_username>
            icon: fab fa-github
          # optionally add other contact information like  
          # - link: <change this> # contact link
          #   icon: <change this> # awsomefont tag for link (check: https://fontawesome.com/v5.15/icons)

    comments:
        # enable comments for your post
        label: false
        kind: comments
    ```
    You can look at the [already merged notebooks](https://github.com/sut-ai/notes/tree/master/notebooks) to find example metadata and folder structures. To learn more about further customizations please consult the [Webifier](https://github.com/webifier/build/) documentations.
5. Add your notebook directory name to the list of notebooks in `notebooks/index.yml`
6. After making sure that you've done everything correctly, proceed to open a pull request with your notebook directory name as the subject.

## License
MIT License, see [notes/LICENSE](https://github.com/sut-ai/notes/blob/master/LICENSE).

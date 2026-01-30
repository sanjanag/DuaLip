
# PyTorch Rewrite - Temporary README

The branch `dualip-pytorch` serves as the master branch while we update the repo
to the new Pytorch version of DuaLip. This branch is protected, and so changes
should be made by creating a new branch from `dualip-pytorch` and then submitting
a pull request.

Once `dualip-pytorch` is stable enough for an initial release, we will set
`dualip-pytorch` to be the master branch. The older scala based master branch
is frozen under the `dualip-legacy-scala` branch. 



# Development setup and workflow

Clone the Github repo and then create a new branch from `dualip-pytorch`. 

1) Create & activate a virtual environment  
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
```

2) Install dev dependencies and pre-commit hooks  
```bash
make install
```

3) Run tests  
```bash
make test
```

4) Run checkstyle (format + lint)  
```bash
make checkstyle
```
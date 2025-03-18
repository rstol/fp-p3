# Dummy Backend

## Installation

Ensure you have Python 3.13 installed. (e.g. with `pyenv install 3.13`)

You can simply install the package through poetry:

```
cd backend
poetry install
```

To generate the source command to your virtual environment enter:

```
poetry env activate
```

To activate the virtual environment run the generated `source ...` command

Then following should return a path to poetry virtualenvs:

```
which python3
```

## How to run

Once the package has been installed, you can run the server by running the `start-server` command directly on your terminal, or by running `python -m server.router.app`.

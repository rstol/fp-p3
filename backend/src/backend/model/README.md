## Training the model

> Before you begin ensure you have [set up the environment variables](../../../README.md#setting-up-basketball_profile) also on the GPU cluster for training

Install the model dependencies with the following in the backend folder:
```
uv sync --group train
```

### Adding the fixed shot clock data

1) Copy `shots_fixed.csv.zip` (which I acquired from [here](https://github.com/sealneaward/nba-movement-data/blob/master/data/shots/shots_fixed.csv) using https://downgit.github.io) to the `TRACKING_DIR` directory and unzip it:

```bash
mkdir -p ${TRACKING_DIR}
cp ${TRACKING_DIR}/shots_fixed.csv.zip ${TRACKING_DIR}
cd ${TRACKING_DIR}
unzip -q shots_fixed.csv.zip
rm shots_fixed.csv.zip
```

This data will be used to determine the offensive team direction of the court 


### Generating the training data

```bash
cd ${PROJECT_DIR}
uv run python preprocess.py --name full
```

> Note: The full dataset will use about 80GB of disk space. The [downloading and extracting (generate splits)](../../../scripts/load_nba_tracking_data_15_16.py) of the ~290'000 plays will take a bit more than 1 hour the first time you run the script. 

You can monitor its progress with:

```bash
ls -U ${GAMES_DIR} | wc -l
```

Then copy the preprocessed files to the GPU cluster instance /scratch folder and copy the files for running the model as well.

## Running the training script

Run the [train model script](../../../scripts/train_model.sh), editing the variables as appropriate.

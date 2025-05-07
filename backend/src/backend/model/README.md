## Training the model

> Before you begin ensure you have [set up the environment variables](../../../README.md#setting-up-basketball_profile) 


### Generating the training data

```bash
cd ${PROJECT_DIR}
nohup uv run python preprocess.py --name full > data.log &
```

> Note: The full dataset will use about 80GB of disk space. The [downloading and extracting (generate splits)](../../../scripts/load_nba_tracking_data_15_16.py) of the ~290'000 plays will take a bit more than 1 hour the first time you run the script. 

You can monitor its progress with:

```bash
ls -U ${GAMES_DIR} | wc -l
```

## Running the training script

Run the [train model script](../../../scripts/train_model.sh), editing the variables as appropriate.

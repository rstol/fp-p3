## Training the model

> Before you begin ensure you have [set up the environment variables](../../../README.md#setting-up-basketball_profile) 

### Adding the events raw metadata

1) Copy `events.zip` (which I acquired from [here](https://github.com/sealneaward/nba-movement-data/tree/master/data/events) \[mirror [here](https://github.com/airalcorn2/nba-movement-data/tree/master/data/events)\] using https://downgit.github.io) to the `EVENTS_DIR` directory and unzip it:

```bash
mkdir -p ${EVENTS_DIR}
cp ${PROJECT_DIR}/events.zip ${EVENTS_DIR}
cd ${EVENTS_DIR}
unzip -q events.zip
rm events.zip
```

Descriptions for the various `EVENTMSGTYPE`s can be found [here](https://github.com/rd11490/NBA_Tutorials/tree/master/analyze_play_by_play) (mirror [here](https://github.com/airalcorn2/NBA_Tutorials/tree/master/analyze_play_by_play)).

### Generating the training data

```bash
cd ${PROJECT_DIR}
nohup uv run python preprocess.py --name full > data.log &
```

You can monitor its progress with:

```bash
ls -U ${GAMES_DIR} | wc -l
```

## Running the training script

Run the [train model script](../../../scripts/train_model.sh), editing the variables as appropriate.

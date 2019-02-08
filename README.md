# Raster Vision Example for LabelImg training

## Using

See [Raster Vision Docs](https://docs.rastervision.io) and [examples repository](https://github.com/azavea/raster-vision-examples) for information about what Raster Vision is and how to run Raster Vision on datasets.

### Step 1: Get the data

Place or link the data in the `data` folder.

### Step 2: Modify your Raster Vision configuration to include the plugin

Make sure to have this in your `~/.rastervision/default` (or `~/.rastervision/PROFILE` if you are using a profile with name PROFILE):

```
[PLUGINS]
modules=["wwt"]
```

This loads the necessary plugins for this project.


### Step 3: Run RasterVision

There's a convenience script that will make running the steps of Raster Vision easier in `scripts/run.py`

Check out the options via `python scripts/run.py --help`, and help for subcommands with e.g. `python scripts/run.py chip --help`.

#### Running stats

You should only try to do  this once; if you already have a `data/rv_root/analyze/default/stats.json`, you won't need to do this. However, if you do some additional preprocessing to the data and change the data in a drastic way (e.g. change everything to grayscale) you'll probably have to run this again. One way to speed things up is to create a directory with a subset of the imagery to create stats over; the sampling should be enough to give statistics that are relevant to the whole dataset.

To run it, do:

```
> python scripts/run.py stats DATA_DIR STATS_ID
```

Where `DATA_DIR` is the data that you want to run stats over (again, can be a subset of data), and `STATS_ID` is how you'll refer to these stats for usage in further commands.

#### Running Chipping

If you want to create training chips from new data, change the chip size existing data, or run data with new stats, you'll need to run the chip command:

```
> python scripts/run.py chip DATA_DIR CHIP_ID --stats-id STATS_ID
```

The `--stats-id STATS_ID` is optional, and only if you're using the stats not found in `data/rv_root/analyze/default/stats.json`.

The `DATA_DIR` contains __all__ of the images you want to create training chips from - the directory can be nested, but each TIF image must have it's XML label file next to it.

The `CHIP_ID` is how you'll refer to these training chips  in further commands.

#### Running Training

To run training and create a new model, run:

```
> python scripts/run.py train DATA_DIR MODEL_ID --stats-id STATS_ID --chip-id
```

This will run the `TRAIN`, `PREDICT`, `EVAL`, and `BUNDLE` Raster Vision commands. The outputs of those commands will be in the `rv_root/{command}/{model_id}` directory.

The `--stats-id STATS_ID` and `--chip-id` are optional, to be used if you want to use specific `stats` and `chip` run results.

The `DATA_DIR` contains the images you want to create training chips from - the directory can be nested, but each TIF image must have it's XML label file next to it.

The `MODEL_ID` is how you'll refer to these training chips  in further commands.

#### Predict on unlabeled imagery

To predict on unlabeled imagery, run:

```
> python scripts/run.py predict INPUT_DIR OUTPUT_DIR --stats-id STATS_ID --chip-id --model-id MODEL_ID
```

The `--stats-id STATS_ID` and `--chip-id` are optional, to be used if you want to use specific `stats` and `chip` run results. The `--model-id` should specificy which model you'd like to use to predict.

`INPUT_DIR` is the directory of unlabeled images you want to predict against. Prediction output will be saved to `OUTPUT_DIR`.

#### Choosing bands

You can choose a different set of 3 bands to use in each of the commands via the `--bands` option. __Note:__ Use the same bands across all commands from chipping to modeling to predicting. So if you have created chips  from bands 4,2,1, you can't train a model off of a different band combination - you'll have to make more chips.

e.g.

```
> python scripts/run.py chip DATA_DIR CHIP_ID --bands 6,3,1
```

Bands are 0-indexed, so the first band is band 0.

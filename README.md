# Raster Vision Exmaple for LabelImg training

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

The Raster Vision command that I'm currently working with is:

```
rastervision -v -p wwt run local -e wwt.chip_classification -a root_uri /opt/data/rv_root
```

Change to your own `root_uri`.

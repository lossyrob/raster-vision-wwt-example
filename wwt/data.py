import os

import pandas as pd
import rastervision as rv

import wwt

def get_scene_configs(data_path, task, bands=None,
                      create_label_source=None):
    """Returns a list of SceneConfigs
    """

    if bands is None:
        bands = [4,2,1]

    def make_scene(img_path, xml_path):
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        if task.task_type == rv.OBJECT_DETECTION:
            label_source = rv.LabelSourceConfig.builder(rv.OBJECT_DETECTION) \
                                               .with_uri(xml_path) \
                                               .build()
        else:
            if create_label_source is None:
                raise Error('You must pass in a "create_label_source" '
                            'function for chip classification')

            label_source = create_label_source(xml_path)

        raster_source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                                             .with_uri(img_path) \
                                             .with_channel_order(bands) \
                                             .with_stats_transformer() \
                                             .build()

        return rv.SceneConfig.builder() \
                             .with_id(img_name) \
                             .with_raster_source(raster_source) \
                             .with_label_source(label_source) \
                             .build()

    scenes = []
    for root, folders, files in os.walk(data_path):
        for f in files:
            img_path = os.path.join(root, f)
            base, ext = os.path.splitext(f)
            if ext.lower() == '.tif':
                xml_path = '{}.xml'.format(os.path.join(root, base))
                if os.path.exists(xml_path):
                    scenes.append(make_scene(img_path, xml_path))
                else:
                    print('WARN: TIFF found without labels: {}'.format(img_path))

    return scenes

def get_predict_scene_configs(data_path, task, bands=None):
    if bands is None:
        bands = [4,2,1]

    def make_scene(img_path):
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        raster_source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                                             .with_uri(img_path) \
                                             .with_channel_order(bands) \
                                             .with_stats_transformer() \
                                             .build()

        return rv.SceneConfig.builder() \
                             .with_id(img_name) \
                             .with_raster_source(raster_source) \
                             .build()


    scenes = []
    for root, folders, files in os.walk(data_path):
        for f in files:
            img_path = os.path.join(root, f)
            base, ext = os.path.splitext(f)
            if ext.lower() == '.tif':
                scenes.append(make_scene(img_path))

    return scenes

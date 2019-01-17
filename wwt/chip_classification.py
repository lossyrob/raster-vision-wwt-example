import os
import sys

import rastervision as rv

from .data import (get_scene_configs,
                   get_predict_scene_configs)

class ChipClassificationExperiments(rv.ExperimentSet):
    def exp_wwt_resnet50_500chip(self,
                                 root_uri,
                                 data_dir,
                                 stats_id='default',
                                 chip_id='default',
                                 model_id='default',
                                 predict_output_dir=None,
                                 bands=None,
                                 test=False):

        CHIP_SIZE = 300

        # # THIS MODEL DOES NOT REQUIRE A CHIP SIZE
        # # IF YOU USE THIS, SET THE "model" PART OF "with_template" BELOW TO READ:
        # #
        # # "model": {
        # #   "input_size": CHIP_SIZE,
        # #   "type": "RESNET50",
        # #   "model_path": "",
        # #   "load_weights_by_name": True,
        # # }
        # #
        # # i.e. set 'load_weights_by_name' to True.
        # PRETRAINED_MODEL = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

        # # THIS MODEL REQUIRES CHIP SIZE 200
        # PRETRAINED_MODEL = ('https://s3.amazonaws.com/azavea-research-public-data/raster-vision/'
        #                     'examples/model-zoo/rio-cc/model-weights.hdf5')


        # THIS MODEL REQUIRES CHIP SIZE 300
        PRETRAINED_MODEL = (
            's3://azavea-research-public-data/models/'
            'cowc-potsdam/classification/resnet50/'
            'model-weights.hdf5')

        IOA_THRESHOLD = 0.5

        if bands is None:
            bands = [4,2,1]
        else:
            bands = list(map(lambda x: int(x), bands.split(',')))

        task = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
                            .with_chip_size(CHIP_SIZE) \
                            .with_classes({
                                "plant": (1, "red"),
                                "no_plant": (2, "black")
                            }) \
                            .build()

        batch_size = 8
        epochs = 40
        if test:
            print("Running test, EPOCHS = 1")
            epochs = 1

        backend = rv.BackendConfig.builder(rv.KERAS_CLASSIFICATION) \
                                  .with_task(task) \
                                  .with_template({
                                      "model": {
                                          "input_size": CHIP_SIZE,
                                          "type": "RESNET50",
                                          "model_path": ""
                                      },
                                      "trainer": {
                                          "optimizer": {
                                              "type": "ADAM",
                                              "init_lr": 0.0001
                                          },
                                          "options": {
                                              "training_data_dir": "",
                                              "validation_data_dir": "",
                                              "nb_epochs": epochs,
                                              "batch_size": batch_size,
                                              "input_size": CHIP_SIZE,
                                              "output_dir": "",
                                              "class_names": ["plant", "no_plant"],
                                              "short_epoch": False
                                          }
                                      }
                                  }) \
                                  .with_pretrained_model(PRETRAINED_MODEL) \
                                  .with_debug(False) \
                                  .with_train_options(replace_model=True) \
                                  .with_batch_size(batch_size) \
                                  .with_num_epochs(epochs) \
                                  .build()

        def create_label_store(uri):
            return rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION) \
                                       .with_uri(uri) \
                                       .with_ioa_thresh(IOA_THRESHOLD) \
                                       .with_use_intersection_over_cell(False) \
                                       .with_pick_min_class_id(True) \
                                       .with_background_class_id(2) \
                                       .with_infer_cells(True) \
                                       .build()

        if predict_output_dir is None:
            scenes = get_scene_configs(data_dir, task, bands, create_label_store)
        else:
            scenes = get_predict_scene_configs(data_dir, task, bands)

        if len(scenes) < 1:
            print('ERROR: There are no scenes generated from {}. '
                  'Check the directory and try again'.format(data_dir))
            sys.exit(1)

        if predict_output_dir is None:
            split_point = int(len(scenes) * 0.8)

            dataset = rv.DatasetConfig.builder() \
                                      .with_train_scenes(scenes[:split_point]) \
                                      .with_validation_scenes(scenes[split_point:]) \
                                      .build()
        else:
            # Required to set a train/validation scene, see GitHub issue #663
            dataset = rv.DatasetConfig.builder() \
                                      .with_train_scene(
                                          rv.SceneConfig.builder() \
                                          .with_id('dummy') \
                                          .with_raster_source(__file__) \
                                          .build()
                                      ) \
                                      .with_validation_scene(
                                          rv.SceneConfig.builder() \
                                          .with_id('dummy') \
                                          .with_raster_source(__file__) \
                                          .build()
                                      ) \
                                      .with_test_scenes(scenes) \
                                      .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id('wwt-chip-classification-{}'.format(model_id)) \
                                        .with_root_uri(root_uri) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .with_stats_analyzer() \
                                        .with_analyze_key(stats_id) \
                                        .with_chip_key(chip_id)

        if predict_output_dir:
            experiment = experiment.with_predict_uri(predict_output_dir)

        return experiment.build()

if __name__ == '__main__':
    rv.main()

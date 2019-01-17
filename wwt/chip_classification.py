import os

import rastervision as rv

from .data import (get_scene_configs, BASE_STATS_JSON_PATH)

class ChipClassificationExperiments(rv.ExperimentSet):
    def exp_wwt_resnet50_500chip(self, root_uri, bands=None, data_uri='/opt/data', test=False):
        if bands is None:
            bands = [4,2,1]
        else:
            bands = list(map(lambda x: int(x), bands.split(',')))

        task = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
                    .with_chip_size(500) \
                    .with_classes({
                        "plant": (1, "red"),
                        "no_plant": (2, "black")
                    }) \
                    .build()

        batch_size = 8
        epochs = 100
        if test:
            print("Running test, EPOCHS = 1")
            epochs = 1

        backend = rv.BackendConfig.builder(rv.KERAS_CLASSIFICATION) \
                                  .with_task(task) \
                                  .with_model_defaults(rv.RESNET50_IMAGENET) \
                                  .with_debug(False) \
                                  .with_train_options(replace_model=True) \
                                  .with_batch_size(batch_size) \
                                  .with_num_epochs(epochs) \
                                  .build()

                                  # .with_config({
                                  #     "trainer": {
                                  #         "options": {
                                  #             "saveBest": True,
                                  #             "lrSchedule": [
                                  #                 {
                                  #                     "epoch": 0,
                                  #                     "lr": 0.0005
                                  #                 },
                                  #                 {
                                  #                     "epoch": 15,
                                  #                     "lr": 0.0001
                                  #                 },
                                  #                 {
                                  #                     "epoch": 30,
                                  #                     "lr": 0.00001
                                  #                 }
                                  #             ]
                                  #         }
                                  #     }
                                  # }, set_missing_keys=True) \


        def create_label_store(uri):
            return rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION) \
                                       .with_uri(uri) \
                                       .with_ioa_thresh(0.5) \
                                       .with_use_intersection_over_cell(False) \
                                       .with_pick_min_class_id(True) \
                                       .with_background_class_id(2) \
                                       .with_infer_cells(True) \
                                       .build()

        scenes = get_scene_configs(data_uri, task, bands, create_label_store)

        split_point = int(len(scenes) * 0.8)

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(scenes[:split_point]) \
                                  .with_validation_scenes(scenes[split_point:]) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id('wwt-chip-classification') \
                                        .with_root_uri(root_uri) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .with_analyze_key("wwt-stats") \
                                        .build()

        return experiment

if __name__ == '__main__':
    rv.main()

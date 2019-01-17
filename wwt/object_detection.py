import os

import rastervision as rv

from .data import get_scene_configs


class ObjectDetectionExperiments(rv.ExperimentSet):

    def exp_wwt(self, root_uri, data_uri='/opt/data', test=False):
        # Number of training steps. Increase this for longer train time
        # and better results.
        NUM_STEPS = 100000
        if test:
            NUM_STEPS = 10

        CHIP_SIZE = 768
        BATCH_SIZE = 4
        BANDS = [4,2,1]

        task = rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                            .with_chip_size(CHIP_SIZE) \
                            .with_classes({"WWTP": (1, "red")}) \
                            .with_chip_options(neg_ratio=1.0,
                                               ioa_thresh=0.8) \
                            .with_predict_options(merge_thresh=0.1,
                                                  score_thresh=0.5) \
                            .build()

        # Set up the backend base.
        # Here we create a builder with the configuration that
        # is common between the two experiments. We don't call
        # build, so that we can branch off the builder based on
        # using a mobilenet or faster rcnn resnet model.

        backend_base = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                                    .with_task(task) \
                                    .with_debug(True) \
                                    .with_batch_size(BATCH_SIZE) \
                                    .with_num_steps(NUM_STEPS) \
                                    .with_train_options(do_monitoring=True,
                                                        replace_model=True)

        mobilenet = backend_base \
                    .with_model_defaults(rv.SSD_MOBILENET_V1_COCO) \
                    .build()

        resnet = backend_base \
                    .with_model_defaults(rv.FASTER_RCNN_RESNET50_COCO) \
                    .build()

        scenes = get_scene_configs(data_uri, task)

        split_point = int(len(scenes) * 0.8)

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(scenes[:split_point]) \
                                  .with_validation_scenes(scenes[split_point:]) \
                                  .build()

        # Set up the experiment base.
        # Notice we set the "chip_key". This allows the two experiments to
        # use the same training chips, so that the chip command is only run
        # once for both experiments.

        experiment_base = rv.ExperimentConfig.builder() \
                                            .with_root_uri(root_uri) \
                                            .with_task(task) \
                                            .with_dataset(dataset) \
                                            .with_chip_key("wwt-object_detection") \
                                            .with_stats_analyzer() \
                                            .with_analyze_key("wwt-stats")

        mn_experiment = experiment_base \
                        .with_id('wwt-object-detection-mobilenet') \
                        .with_backend(mobilenet) \
                        .build()

        rn_experiment = experiment_base \
                        .with_id('wwt-object-detection-resnet') \
                        .with_backend(resnet) \
                        .build()

        return [mn_experiment, rn_experiment]

if __name__ == '__main__':
    rv.main()

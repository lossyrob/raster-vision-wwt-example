import rastervision as rv

import shipdetect as sd
from shipdetect.data import get_scene_configs

class ShipDetectExperiments(rv.ExperimentSet):
    def exp_deeplab(self, rv_root, raw_root, test_run=False):
        """Run an experiment using deeplab.

        Args:
           rv_root - The root URI where experiment output will be held.
           raw_root - The root URI where training data is located.
        """


        class_map = {
            'ship': (1, 'yellow'),
            'background': (2, 'black')
        }

        model_type = rv.MOBILENET_V2
        # model_type = rv.XCEPTION_65

        if test_run:
            num_steps = 10
        else:
            num_steps = 50000

        batch_size = 8

        experiment_id = "deeplab_{}-{}batch_{}steps".format(model_type, batch_size, num_steps)

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_classes(class_map) \
                            .with_chip_size(384) \
                            .with_chip_options(target_classes=[1],
                                               chips_per_scene=10) \
                            .build()

        backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                                  .with_task(task) \
                                  .with_model_defaults(model_type) \
                                  .with_num_steps(num_steps) \
                                  .with_batch_size(batch_size) \
                                  .with_debug(True) \
                                  .build()

        print("Loading scenes...", end='', flush=True)
        scenes = get_scene_configs('/opt/data/train_ship_segmentations_v2.csv',
                                   task,
                                   raw_root)
        print("done.")

        if test_run:
            scenes = scenes[:5]

        scene_count = len(scenes)
        split_point = int(scene_count * 0.8)
        train_scenes = map(lambda x: x[0], scenes[:split_point])
        val_scenes = map(lambda x: x[0], scenes[split_point:])

        print("Building dataset ({:,} train scenes - {:,} val scenes)".format(
            split_point, scene_count - split_point))
        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id(experiment_id) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .with_root_uri(rv_root) \
                                        .build()

        return experiment

if __name__ == '__main__':
    rv.main()

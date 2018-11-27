import os
from copy import deepcopy

import rastervision as rv
from rastervision.data.vector_source import (VectorSource,
                                             VectorSourceConfig,
                                             VectorSourceConfigBuilder)
from rastervision.data.vector_source.default import VectorSourceDefaultProvider
from rastervision.data.vector_source.class_inference import ClassInferenceOptions

import wwt
from wwt.utils import label_img_file_to_geojson

class LabelImgVectorSource(VectorSource):
    def __init__(self, uri, crs_transformer, extent, class_inf_opts=None):
        """Constructor.

        Args:
            uri: (str) uri of GeoJSON file
            crs_transformer: (CRSTransformer)
            extent: (Box) extent of scene which determines which features to return
            class_inf_opts: ClassInferenceOptions
        """
        self.uri = uri
        self.crs_transformer = crs_transformer
        self.extent = extent
        super().__init__(class_inf_opts)

    def _get_geojson(self):
        """Converts from LabelImg XML to a GeoJSON dict"""
        return label_img_file_to_geojson(self.uri, self.extent, self.crs_transformer)


class LabelImgVectorSourceConfig(VectorSourceConfig):
    def __init__(self, uri, class_id_to_filter=None, default_class_id=1):
        self.uri = uri
        super().__init__(
            wwt.LABEL_IMG_SOURCE,
            class_id_to_filter=class_id_to_filter,
            default_class_id=default_class_id)

    def to_proto(self):
        msg = super().to_proto()
        msg.geojson.uri = self.uri
        return msg

    def create_source(self, crs_transformer=None, extent=None, class_map=None):
        return LabelImgVectorSource(
            self.uri,
            crs_transformer,
            extent,
            class_inf_opts=ClassInferenceOptions(
                class_map=class_map,
                class_id_to_filter=self.class_id_to_filter,
                default_class_id=self.default_class_id))

    def update_for_command(self,
                           command_type,
                           experiment_config,
                           context=None,
                           io_def=None):
        io_def = io_def or rv.core.CommandIODefinition()
        io_def.add_input(self.uri)
        return io_def


class LabelImgVectorSourceConfigBuilder(VectorSourceConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'uri': prev.uri,
                'class_id_to_filter': prev.class_id_to_filter,
                'default_class_id': prev.default_class_id
            }

        super().__init__(LabelImgVectorSourceConfig, config)

    def validate(self):
        if self.config.get('uri') is None:
            raise rv.ConfigError(
                'LabelImgVectorSourceConfigBuilder requires uri which '
                'can be set using "with_uri".')

        super().validate()

    def from_proto(self, msg):
        b = super().from_proto(msg)
        b = b.with_uri(msg.geojson.uri)
        return b

    def with_uri(self, uri):
        b = deepcopy(self)
        b.config['uri'] = uri
        return b


class LabelImgVectorSourceDefaultProvider(VectorSourceDefaultProvider):
    @staticmethod
    def handles(uri):
        ext = os.path.splitext(uri)[1]
        return ext.lower() == '.xml'

    @staticmethod
    def construct(uri):
        return rv.VectorSourceConfig.builder(wwt.LABEL_IMG_SOURCE) \
                                    .with_uri(uri) \
                                    .build()

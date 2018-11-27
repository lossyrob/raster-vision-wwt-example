import rastervision as rv

from wwt.vector_source import (LabelImgVectorSourceConfigBuilder,
                               LabelImgVectorSourceDefaultProvider)

LABEL_IMG_SOURCE = 'LABEL_IMG_SOURCE'

def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(rv.VECTOR_SOURCE, LABEL_IMG_SOURCE,
                                            LabelImgVectorSourceConfigBuilder)

    plugin_registry.register_default_vector_source(LabelImgVectorSourceDefaultProvider)

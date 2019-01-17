import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError

from rastervision.utils.files import file_to_str

def label_img_object_to_feature(o):
    """Converts an ElementTree element representing an "object" element
    in LabelImg XML into a GeoJSON Feature"""
    props = { 'class_name': o.find('name').text }

    bbox = o.find('bndbox')
    xmin = float(bbox.find('xmin').text)
    ymin = float(bbox.find('ymin').text)
    xmax = float(bbox.find('xmax').text)
    ymax = float(bbox.find('ymax').text)

    geom = {
        'type': 'Polygon',
        'coordinates': [
            [
                [xmin, ymax],
                [xmax, ymax],
                [xmax, ymin],
                [xmin, ymin],
                [xmin, ymax]
            ]
        ]
    }

    return {
        'type': 'Feature',
        'properties': props,
        'geometry': geom
    }

def label_img_xml_to_geojson(xml, extent, crs_transformer):
    """Converts an ElementTree root element representing
    LabelImg XML labels to a GeoJSON FeatureCollection dict"""

    cols = float(xml.find('size').find('width').text)
    rows = float(xml.find('size').find('height').text)

    cell_width = extent.get_width() / cols
    cell_height = extent.get_height() / rows

    def convert_feature_to_latlng(feature):
        new_ring = []

        for coord in feature['geometry']['coordinates'][0]:
            col, row = coord[0], coord[1]
            x, y = crs_transformer.pixel_to_map((col, row))

            new_ring.append([x, y])

        return {
            'type': feature['type'],
            'properties': feature['properties'],
            'geometry': {
                'type': 'Polygon',
                'coordinates': [ new_ring ]
            }
        }

    features = [
        convert_feature_to_latlng(
            label_img_object_to_feature(o)
        ) for o in xml.findall('object')
    ]

    return {
        'type': 'FeatureCollection',
        'features': features
    }

def label_img_str_to_geojson(s, extent, crs_transformer):
    return label_img_xml_to_geojson(ET.fromstring(s), extent, crs_transformer)



def label_img_file_to_geojson(uri, extent, crs_transformer):
    try:
        return label_img_str_to_geojson(file_to_str(uri), extent, crs_transformer)
    except ParseError:
        print('WARNING: Cannot parse {}'.format(uri))
        return {
            'type': 'FeatureCollection',
            'features': []
        }

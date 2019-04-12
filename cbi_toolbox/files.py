import bioformats
import javabridge
import json
import numpy as np
import os


def load_mm_ome_tiff(file_path):
    javabridge.start_vm(class_path=bioformats.JARS)

    image_reader = bioformats.formatreader.make_image_reader_class()()
    format_tools = bioformats.formatreader.make_format_tools_class()
    image_reader.allowOpenToCheckType(True)
    image_reader.setId(file_path)

    size_x = image_reader.getSizeX()
    size_y = image_reader.getSizeY()
    size_c = image_reader.getSizeC()
    size_z = image_reader.getSizeZ()
    size_t = image_reader.getSizeT()

    pixel_type = format_tools.getPixelTypeString(image_reader.getPixelType())

    ome_tiff_array = np.empty((size_x, size_y, size_c, size_z, size_t), dtype=pixel_type)
    with bioformats.ImageReader(file_path, perform_init=True) as reader:
        reader.rdr = image_reader

        for channel in range(size_c):
            for z_index in range(size_z):
                for time in range(size_t):
                    ome_tiff_array[:, :, channel, z_index, time] = reader.read(c=channel, z=z_index, t=time,
                                                                               rescale=False)

    javabridge.kill_vm()

    file_name = os.path.splitext(os.path.splitext(file_path)[0])[0]
    metadata_path = '_'.join((file_name, 'metadata.txt'))

    with open(metadata_path) as f:
        metadata_json = json.load(f)

    return ome_tiff_array, metadata_json

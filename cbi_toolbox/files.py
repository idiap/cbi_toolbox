import bioformats
import javabridge
import json
import numpy as np
import os


def load_mm_ome_tiff(file_path, kill_vm=False):
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

    if kill_vm:
        javabridge.kill_vm()

    file_name = os.path.splitext(os.path.splitext(file_path)[0])[0]
    metadata_path = '_'.join((file_name, 'metadata.txt'))

    with open(metadata_path) as f:
        metadata_json = json.load(f)

    return ome_tiff_array, metadata_json


def save_to_ome(image5d, out_file, file_name_for_metadata=None, resolutions=np.array([]), units=np.array([]),
                pixel_type='float'):
    """
    :param image5d: 5D ndarray. Expected XYCZT order
    :param out_file: /path/to/output.ome Can be any bioformat supported format. Tested for OME and OMETIFF so far
    :param file_name_for_metadata: Will copy the physical sizes metadata of this file to the output file
    :param resolutions: if no metadata, the physical size can be inserted here
    :param units: if no metadata, the physical size units can be inserted here
    :param pixel_type: if no metadata, the pixel type an be inserted here
    :return: no return. The 5D ndarray is saved to the output location
    """

    image_writer = bioformats.formatwriter.make_image_writer_class()
    writer = image_writer()

    if file_name_for_metadata is not None:
        rdr = bioformats.ImageReader(file_name_for_metadata, perform_init=True)
        jmd = javabridge.JWrapper(rdr.rdr.getMetadataStore())
        pixel_type = jmd.getPixelsType(0).getValue()

        omexml = bioformats.OMEXML()
        omexml.image(0).Name = os.path.split(out_file)[1]
        p = omexml.image(0).Pixels
        assert isinstance(p, bioformats.OMEXML.Pixels)

        p.node.set("PhysicalSizeX", str(jmd.getPixelsPhysicalSizeX(0).value()))
        p.node.set("PhysicalSizeY", str(jmd.getPixelsPhysicalSizeY(0).value()))
        p.node.set("PhysicalSizeZ", str(jmd.getPixelsPhysicalSizeZ(0).value()))
        p.node.set("PhysicalSizeXUnit", jmd.getPixelsPhysicalSizeX(0).unit().getSymbol())
        p.node.set("PhysicalSizeYUnit", jmd.getPixelsPhysicalSizeY(0).unit().getSymbol())
        p.node.set("PhysicalSizeZUnit", jmd.getPixelsPhysicalSizeZ(0).unit().getSymbol())

        p.SizeX = image5d.shape[1]
        p.SizeY = image5d.shape[0]
        p.SizeC = image5d.shape[2]
        p.SizeT = image5d.shape[4]
        p.SizeZ = image5d.shape[3]

        p.DimensionOrder = bioformats.omexml.DO_XYCZT
        p.PixelType = pixel_type

    else:
        resolutions = np.array(resolutions)
        units = np.array(units)
        omexml = bioformats.OMEXML()
        omexml.image(0).Name = os.path.split(out_file)[1]
        p = omexml.image(0).Pixels
        assert isinstance(p, bioformats.OMEXML.Pixels)
        if resolutions.size:
            Rx, Ry, Rz = resolutions
            p.node.set("PhysicalSizeX", str(Rx))
            p.node.set("PhysicalSizeY", str(Ry))
            p.node.set("PhysicalSizeZ", str(Rz))
        if units.size:
            Ux, Uy, Uz = units
            p.node.set("PhysicalSizeXUnit", Ux)
            p.node.set("PhysicalSizeYUnit", Uy)
            p.node.set("PhysicalSizeZUnit", Uz)
        p.SizeX = image5d.shape[1]
        p.SizeY = image5d.shape[0]
        p.SizeC = image5d.shape[2]
        p.SizeZ = image5d.shape[3]
        p.SizeT = image5d.shape[4]
        p.DimensionOrder = bioformats.omexml.DO_XYCZT
        p.PixelType = pixel_type

        if image5d.ndim == 3:
            p.SizeC = image5d.shape[2]
            p.Channel(0).SamplesPerPixel = image5d.shape[2]
            omexml.structured_annotations.add_original_metadata(
                bioformats.omexml.OM_SAMPLES_PER_PIXEL, str(image5d.shape[2]))
        elif image5d.shape[2] > 1:
            p.channel_count = image5d.shape[2]

    xml = omexml.to_xml()
    script = """
    importClass(Packages.loci.formats.services.OMEXMLService,
                Packages.loci.common.services.ServiceFactory,
                Packages.loci.formats.MetadataTools,
                Packages.loci.formats.meta.IMetadata);

    service = new ServiceFactory().getInstance(OMEXMLService);
    metadata = service.createOMEXMLMetadata(xml);

    """
    meta = javabridge.run_script(script, dict(xml=xml))
    if os.path.isfile(out_file):
        os.remove(out_file)

    writer.setMetadataRetrieve(meta)
    writer.setId(out_file)

    nc, nz, nt = image5d.shape[2:]

    for t in range(nt):
        for c in range(nc):
            for z in range(nz):
                index = c + nc * z + nc * nz * t
                save_im = bioformats.formatwriter.convert_pixels_to_buffer(image5d[..., c, z, t], p.PixelType)
                writer.saveBytesIB(index, save_im)
    writer.close()


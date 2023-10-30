"""
Methods for creating a variety of SIDD products.

Examples
--------
Create a variety of sidd products.

.. code-block:: python

    import os

    from sarpy.io.complex.converter import open_complex
    from sarpy.processing.ortho_rectify import BivariateSplineMethod, NearestNeighborMethod
    from sarpy.processing.sidd.sidd_product_creation import create_detected_image_sidd, create_csi_sidd, create_dynamic_image_sidd

    # open a sicd type file
    reader = open_complex('<sicd type object file name>')
    # create an orthorectification helper for specified sicd index
    ortho_helper = NearestNeighborMethod(reader, index=0)

    # create a sidd version 2 detected image for the whole file
    create_detected_image_sidd(ortho_helper, '<output directory>', block_size=10, version=2)
    # create a sidd version 2 color sub-aperture image for the whole file
    create_csi_sidd(ortho_helper, '<output directory>', dimension=0, version=2)
    # create a sidd version 2 dynamic image/sub-aperture stack for the whole file
    create_dynamic_image_sidd(ortho_helper, '<output directory>', dimension=0, version=2)
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import copy
import inspect
import multiprocessing as mp
import os

from sarpy.io.complex.converter import open_complex
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.product.sidd import SIDDWriter
from sarpy.io.general.base import SarpyIOError
from sarpy.processing.ortho_rectify import BivariateSplineMethod, NearestNeighborMethod
from sarpy.processing.ortho_rectify.base import FullResolutionFetcher, OrthorectificationIterator
from sarpy.processing.ortho_rectify.ortho_methods import OrthorectificationHelper
from sarpy.processing.sidd.sidd_structure_creation import create_sidd_structure
from sarpy.processing.sicd.csi import CSICalculator
from sarpy.processing.sicd.subaperture import SubapertureCalculator, SubapertureOrthoIterator
from sarpy.visualization.remap import MonochromaticRemap, NRL
import sarpy.visualization.remap as remap

# TODO: move this to processing for 1.3.0

DEFAULT_IMG_REMAP = NRL
DEFAULT_CSI_REMAP = NRL
DEFAULT_DI_REMAP = NRL

_output_text = 'output_directory `{}`\n\t' \
               'does not exist or is not a directory'
_orthohelper_text = 'ortho_helper is required to be an instance of OrthorectificationHelper,\n\t' \
                    'got type `{}`'


def _validate_filename(output_directory, output_file, sidd_structure):
    """
    Validate the output filename.

    Parameters
    ----------
    output_directory : str
    output_file : None|str
    sidd_structure

    Returns
    -------
    str
    """

    if output_file is None:
        # noinspection PyProtectedMember
        fstem = os.path.split(sidd_structure.NITF['SUGGESTED_NAME']+'.nitf')[1]
    else:
        fstem = os.path.split(output_file)[1]

    full_filename = os.path.join(os.path.expanduser(output_directory), fstem)
    if os.path.exists(full_filename):
        raise SarpyIOError('File {} already exists.'.format(full_filename))
    return full_filename


def _validate_remap_function(remap_function):
    """
    Verify that the given monochromatic remap function is viable for SIDD
    production.

    Parameters
    ----------
    remap_function : MonochromaticRemap
    """

    if not isinstance(remap_function, MonochromaticRemap):
        raise TypeError('remap_function must be an instance of MonochromaticRemap')
    if remap_function.bit_depth not in [8, 16]:
        raise TypeError('remap_function usage for SIDD requires 8 or 16 bit output')

# Some versions of numpy/scipy (e.g., Anaconda packages that use mkl) change the process affinity
# of the current process to CPU0, only.  This defeats the whole point of multiprocessing.
# Building an environment that avoids problematic packages is the best solution to the problem.
# However, it can be hard to know whether a runtime environment has the problem or not.
# The CPUAffinity class is used within _mp_worker subprocesses to change the CPU affinity back
# to all CPUs, just in case we are using a problematic package.
class CPUAffinity():
    """
    This class will get | set | save | restore the CPU affinity of the current process.
    It is esentially a wrapper for the os.sched_getaffinity and os.sched_setaffinity functions,
    but since the documentation claims that these function are not available on all systems,
    we protect the function calls with try-except blocks.  On systems that do not have
    os.sched_getaffinity and os.sched_setaffinity functions, this class does nothing.
    """

    __slots__ = ("_saved_affinity",)

    def __init__(self, new_affinity=None):
        """
        Create an CPUAffinity object initialized to the current affinity
        unless new_affinity is used to provide a new affinity value.

        Args
        ----
        new_affinity: set | None
            Optional argument used to specify a new set of affinity values.

        """
        if new_affinity:
            self.set(new_affinity)
        self.save()

    def save(self):
        """Save to current affinity"""
        self._saved_affinity = self.get()

    def restore(self):
        """Restore the affinity to a previously saved value"""
        if self._saved_affinity is not None:
            self.set(self._saved_affinity)

    def set(self, new_affinity):
        """ Set the affinity to a user specified value.

        Args
        ----
        new_affinity: set
            A set containing the specified CPU numbers
        """
        try:
            os.sched_setaffinity(0, set(new_affinity))
        except AttributeError:
            pass

    def get(self):
        """Get the current affinity set."""
        try:
            affinity = os.sched_getaffinity(0)
        except AttributeError:
            affinity = None
        return affinity


def _mp_worker(pars):
    affinity_obj = CPUAffinity()        # Capture the CPU affinity prior to execution

    with open_complex(pars['sicd_filename']) as reader:
        # Create an ortho_helper object
        if pars['ortho_helper']['_obj_type_'].endswith('BivariateSplineMethod'):
            ortho_helper = BivariateSplineMethod(reader,
                                                 index=pars['ortho_helper']['index'],
                                                 row_order=pars['ortho_helper']['row_order'],
                                                 col_order=pars['ortho_helper']['col_order'])
        elif pars['ortho_helper']['_obj_type_'].endswith('NearestNeighborMethod'):
            ortho_helper = NearestNeighborMethod(reader,
                                                 index=pars['ortho_helper']['index'])
        else:
            raise ValueError(f"Unknown ortho_helper object type: {pars['ortho_helper']['_obj_type_']}")

        affinity_obj.restore()          # Restore the affinity in case it has changed

        # Create a remap_function object
        if pars['remap_function']['_obj_type_'].endswith('Density'):
            remap_function = remap.Density(override_name=pars['remap_function']['override_name'],
                                           bit_depth=pars['remap_function']['bit_depth'],
                                           max_output_value=pars['remap_function']['max_output_value'],
                                           data_mean=pars['remap_function']['data_mean'])
        elif pars['remap_function']['_obj_type_'].endswith('High_Contrast'):
            remap_function = remap.High_Contrast(override_name=pars['remap_function']['override_name'],
                                                 bit_depth=pars['remap_function']['bit_depth'],
                                                 max_output_value=pars['remap_function']['max_output_value'],
                                                 data_mean=pars['remap_function']['data_mean'])
        elif pars['remap_function']['_obj_type_'].endswith('Brighter'):
            remap_function = remap.Brighter(override_name=pars['remap_function']['override_name'],
                                            bit_depth=pars['remap_function']['bit_depth'],
                                            max_output_value=pars['remap_function']['max_output_value'],
                                            data_mean=pars['remap_function']['data_mean'])
        elif pars['remap_function']['_obj_type_'].endswith('Darker'):
            remap_function = remap.Darker(override_name=pars['remap_function']['override_name'],
                                          bit_depth=pars['remap_function']['bit_depth'],
                                          max_output_value=pars['remap_function']['max_output_value'],
                                          data_mean=pars['remap_function']['data_mean'])
        elif pars['remap_function']['_obj_type_'].endswith('PEDF'):
            remap_function = remap.PEDF(override_name=pars['remap_function']['override_name'],
                                        bit_depth=pars['remap_function']['bit_depth'],
                                        max_output_value=pars['remap_function']['max_output_value'],
                                        data_mean=pars['remap_function']['data_mean'])
        elif pars['remap_function']['_obj_type_'].endswith('GDM'):
            remap_function = remap.GDM(override_name=pars['remap_function']['override_name'],
                                       bit_depth=pars['remap_function']['bit_depth'],
                                       max_output_value=pars['remap_function']['max_output_value'],
                                       weighting=pars['remap_function']['weighting'],
                                       graze_deg=pars['remap_function']['graze_deg'],
                                       slope_deg=pars['remap_function']['slope_deg'],
                                       data_mean=pars['remap_function']['data_mean'],
                                       data_median=pars['remap_function']['data_median'])
        elif pars['remap_function']['_obj_type_'].endswith('Linear'):
            remap_function = remap.Linear(override_name=pars['remap_function']['override_name'],
                                          bit_depth=pars['remap_function']['bit_depth'],
                                          max_output_value=pars['remap_function']['max_output_value'],
                                          min_value=pars['remap_function']['min_value'],
                                          max_value=pars['remap_function']['max_value'])
        elif pars['remap_function']['_obj_type_'].endswith('Logarithmic'):
            remap_function = remap.Logarithmic(override_name=pars['remap_function']['override_name'],
                                               bit_depth=pars['remap_function']['bit_depth'],
                                               max_output_value=pars['remap_function']['max_output_value'],
                                               min_value=pars['remap_function']['min_value'],
                                               max_value=pars['remap_function']['max_value'])
        elif pars['remap_function']['_obj_type_'].endswith('NRL'):
            remap_function = remap.NRL( override_name=pars['remap_function']['override_name'],
                                        bit_depth=pars['remap_function']['bit_depth'],
                                        max_output_value=pars['remap_function']['max_output_value'],
                                        knee=pars['remap_function']['knee'],
                                        percentile=pars['remap_function']['percentile'],
                                        stats=pars['remap_function']['stats'])
        else:
            raise ValueError(f"Unknown remap_function object type: {pars['remap_function']['_obj_type_']}")

        affinity_obj.restore()          # Restore the affinity in case it has changed

        # Create a calculator object
        if pars['calculator']['_obj_type_'].endswith('FullResolutionFetcher'):
            calculator = FullResolutionFetcher(reader,
                                               dimension=pars['calculator']['dimension'],
                                               index=pars['calculator']['index'],
                                               block_size=pars['calculator']['block_size'])

        elif pars['calculator']['_obj_type_'].endswith('CSICalculator'):
            calculator = CSICalculator(reader,
                                       dimension=pars['calculator']['dimension'],
                                       index=pars['calculator']['index'],
                                       block_size=pars['calculator']['block_size'])

        elif pars['calculator']['_obj_type_'].endswith('SubapertureCalculator'):
            calculator = SubapertureCalculator(reader,
                                               dimension=pars['calculator']['dimension'],
                                               index=pars['calculator']['index'],
                                               block_size=pars['calculator']['block_size'],
                                               frame_count=pars['calculator']['frame_count'],
                                               aperture_fraction=pars['calculator']['aperture_fraction'],
                                               method=pars['calculator']['method'])

        else:
            raise ValueError(f"Unknown calculator object type: {pars['calculator']['_obj_type_']}")

        affinity_obj.restore()          # Restore the affinity in case it has changed

        results = []
        if pars['_obj_type_'].endswith('OrthorectificationIterator'):
            # Create an OrthorectificationIterator object
            ortho_iterator = OrthorectificationIterator(ortho_helper,
                                                        calculator=calculator,
                                                        bounds=pars['bounds'],
                                                        remap_function=remap_function,
                                                        recalc_remap_globals=pars['recalc_remap_globals'])
            ortho_iterator._iteration_blocks = pars['_iteration_blocks']

            affinity_obj.restore()          # Restore the affinity in case it has changed

            # Iterate the ortho_iterator object to generate the SIDD data and start index values
            for data, start_indices in ortho_iterator:
                affinity_obj.restore()      # Restore the affinity in case it has changed
                results.append((data, start_indices))

        elif pars['_obj_type_'].endswith('SubapertureOrthoIterator'):
            # Create a SubapertureOrthoIterator object
            ortho_iterator = SubapertureOrthoIterator(ortho_helper,
                                                      calculator=calculator,
                                                      bounds=pars['bounds'],
                                                      remap_function=remap_function,
                                                      recalc_remap_globals=pars['recalc_remap_globals'],
                                                      depth_first=pars['depth_first'])
            ortho_iterator._iteration_blocks = pars['_iteration_blocks']

            affinity_obj.restore()          # Restore the affinity in case it has changed

            # Iterate the ortho_iterator object to generate the SIDD data and start index values
            for data, start_indices, the_frame in ortho_iterator:
                affinity_obj.restore()      # Restore the affinity in case it has changed
                results.append((data, start_indices, the_frame))
        else:
            raise ValueError(f"Unknown OrthoIterator object type: {pars['_obj_type_']}")

    return results


def _extract_args_from_obj(obj):
    obj_args = list(inspect.getfullargspec(obj.__init__))[0][1:]
    obj_pars = {name: obj.__getattribute__(f'_{name}') for name in obj_args if f'_{name}' in obj.__dir__()}
    obj_pars['_obj_type_'] = str(obj).split()[0].split()[0][1:]

    for key, val in obj_pars.items():
        if isinstance(val, SICDType):
            # Omit SICDType objects
            obj_pars[key] = None
            continue

        if str(val).startswith('<') and str(val).endswith('>'):
            if 'object' in str(val):
                # This is a class object, so recursively get its __init__ parameters
                obj_pars[key] = _extract_args_from_obj(val)
            else:
                # We don't know what this is.  Hopefully, we don't need it.
                obj_pars[key] = None

    return obj_pars


def _make_multiproc_iter(ortho_iterator, number_of_processes):
    """Extract the parameters used to create an ortho_iterator and use them to make a multiproc iterator"""
    interator_pars = _extract_args_from_obj(ortho_iterator)
    interator_pars['sicd_filename'] = ortho_iterator.ortho_helper.reader.file_name

    # Determine how to spread the work between the worker processes
    num_blocks_total = len(ortho_iterator._iteration_blocks)
    num_blocks_min_cpu = num_blocks_total // number_of_processes
    num_blocks_each_cpu = [num_blocks_min_cpu] * number_of_processes
    for i in range(num_blocks_total % number_of_processes):
        num_blocks_each_cpu[i] += 1

    # Create a list of parameter sets, one for each worker process
    start_block = 0
    proc_pars = []
    for i, num_block_this_cpu in enumerate(num_blocks_each_cpu):
        if num_block_this_cpu:
            this_iter_pars = copy.deepcopy(interator_pars)
            end_block = start_block + num_block_this_cpu
            this_iter_pars['_iteration_blocks'] = ortho_iterator._iteration_blocks[start_block:end_block]
            this_iter_pars['_worker_id'] = i
            start_block = end_block
            proc_pars.append(this_iter_pars)

    # Convert the list of parameter sets into an iterator usable by multiprocessing.imap.
    multiproc_iter = iter(proc_pars)

    return multiproc_iter


def _validate_number_of_processes(candidate_number_of_processes):
    return mp.cpu_count() if candidate_number_of_processes < 1 else candidate_number_of_processes


def create_detected_image_sidd(
        ortho_helper, output_directory, output_file=None, block_size=10, dimension=0,
        bounds=None, version=3, include_sicd=True, remap_function=None, number_of_processes=1):
    """
    Create a SIDD version of a basic detected image from a SICD type reader.

    Parameters
    ----------
    ortho_helper : OrthorectificationHelper
        The ortho-rectification helper object.
    output_directory : str
        The output directory for the given file.
    output_file : None|str
        The file name, this will default to a sensible value.
    block_size : int
        The approximate processing block size to fetch, given in MB. The
        minimum value for use here will be 1.
    dimension : int
        Which dimension to split over in block processing? Must be either 0 or 1.
    bounds : None|numpy.ndarray|list|tuple
        The sicd pixel bounds of the form `(min row, max row, min col, max col)`.
        This will default to the full image.
    version : int
        The SIDD version to use, must be one of 1, 2, or 3.
    include_sicd : bool
        Include the SICD structure in the SIDD file?
    remap_function : None|MonochromaticRemap
        The applied remap function. If one is not provided, then a default is
        used. Required global parameters will be calculated if they are missing,
        so the internal state of this remap function may be modified.
    number_of_processes: int
       The number of worker processes to use (default = 1).
       If number_of_processes = 0 then the number returned by os.cpu_count() is used.

    Returns
    -------
    None

    Examples
    --------
    .. code-block:: python

        import os

        from sarpy.io.complex.converter import open_complex
        from sarpy.processing.ortho_rectify import BivariateSplineMethod, NearestNeighborMethod
        from sarpy.processing.sidd.sidd_product_creation import create_detected_image_sidd

        reader = open_complex('<sicd type object file name>')
        ortho_helper = NearestNeighborMethod(reader, index=0)

        # create a sidd version 2 file for the whole file
        create_detected_image_sidd(ortho_helper, '<output directory>', block_size=10, version=2)
    """
    number_of_processes = _validate_number_of_processes(number_of_processes)

    if not os.path.isdir(output_directory):
        raise SarpyIOError(_output_text.format(output_directory))

    if not isinstance(ortho_helper, OrthorectificationHelper):
        raise TypeError(_orthohelper_text.format(type(ortho_helper)))

    if remap_function is None:
        remap_function = DEFAULT_IMG_REMAP(override_name='IMG_DEFAULT')        # pragma: no cover
    _validate_remap_function(remap_function)

    # construct the ortho-rectification iterator - for a basic data fetcher
    calculator = FullResolutionFetcher(
        ortho_helper.reader, dimension=dimension, index=ortho_helper.index, block_size=block_size)
    ortho_iterator = OrthorectificationIterator(
        ortho_helper, calculator=calculator, bounds=bounds,
        remap_function=remap_function, recalc_remap_globals=False)

    # create the sidd structure
    ortho_bounds = ortho_iterator.ortho_bounds
    sidd_structure = create_sidd_structure(
        ortho_helper, ortho_bounds,
        product_class='Detected Image', pixel_type='MONO{}I'.format(remap_function.bit_depth), version=version)
    # set suggested name
    sidd_structure.NITF['SUGGESTED_NAME'] = ortho_helper.sicd.get_suggested_name(ortho_helper.index)+'_IMG'

    full_filename = _validate_filename(output_directory, output_file, sidd_structure)

    if number_of_processes > 1:
        multiproc_iter = _make_multiproc_iter(ortho_iterator, number_of_processes)
        with mp.Pool(processes=number_of_processes) as pool:
            results_interators = pool.imap_unordered(_mp_worker, multiproc_iter)
            pool.close()
            pool.join()
    else:
        results_interators = [ortho_iterator]

    with SIDDWriter(full_filename, sidd_structure, ortho_helper.sicd if include_sicd else None) as writer:
        for results in results_interators:
            for data, start_indices in results:
                writer(data, start_indices=start_indices, index=0)


def create_csi_sidd(
        ortho_helper, output_directory, output_file=None, dimension=0,
        block_size=30, bounds=None, version=3, include_sicd=True, remap_function=None, number_of_processes=1):
    """
    Create a SIDD version of a Color Sub-Aperture Image from a SICD type reader.

    Parameters
    ----------
    ortho_helper : OrthorectificationHelper
        The ortho-rectification helper object.
    output_directory : str
        The output directory for the given file.
    output_file : None|str
        The file name, this will default to a sensible value.
    dimension : int
        The dimension over which to split the sub-aperture.
    block_size : int
        The approximate processing block size to fetch, given in MB. The
        minimum value for use here will be 1.
    bounds : None|numpy.ndarray|list|tuple
        The sicd pixel bounds of the form `(min row, max row, min col, max col)`.
        This will default to the full image.
    version : int
        The SIDD version to use, must be one of 1, 2, or 3.
    include_sicd : bool
        Include the SICD structure in the SIDD file?
    remap_function : None|MonochromaticRemap
        The applied remap function. For csi processing, this must explicitly be
        an 8-bit remap. If one is not provided, then a default is used. Required
        global parameters will be calculated if they are missing, so the internal
        state of this remap function may be modified.
    number_of_processes: int
       The number of worker processes to use (default = 1).
       If number_of_processes = 0 then the number returned by os.cpu_count() is used.

    Returns
    -------
    None

    Examples
    --------
    .. code-block:: python

        import os
        from sarpy.io.complex.converter import open_complex
        from sarpy.processing.sidd.sidd_product_creation import create_csi_sidd
        from sarpy.processing.sicd.csi import CSICalculator
        from sarpy.processing.ortho_rectify import NearestNeighborMethod

        reader = open_complex('<sicd type object file name>')
        ortho_helper = NearestNeighborMethod(reader, index=0)
        create_csi_sidd(ortho_helper, '<output directory>', dimension=0, version=2)

    """
    number_of_processes = _validate_number_of_processes(number_of_processes)

    if not os.path.isdir(output_directory):
        raise SarpyIOError(_output_text.format(output_directory))

    if not isinstance(ortho_helper, OrthorectificationHelper):
        raise TypeError(_orthohelper_text.format(type(ortho_helper)))

    # construct the CSI calculator class
    csi_calculator = CSICalculator(
        ortho_helper.reader, dimension=dimension, index=ortho_helper.index, block_size=block_size)

    if remap_function is None:
        remap_function = DEFAULT_CSI_REMAP(override_name='CSI_DEFAULT', bit_depth=8)        # pragma: no cover
    _validate_remap_function(remap_function)
    if remap_function.bit_depth != 8:
        raise ValueError('The CSI SIDD specifically requires an 8-bit remap function.')     # pragma: no cover

    # construct the ortho-rectification iterator
    ortho_iterator = OrthorectificationIterator(
        ortho_helper, calculator=csi_calculator, bounds=bounds,
        remap_function=remap_function, recalc_remap_globals=False)

    # create the sidd structure
    ortho_bounds = ortho_iterator.ortho_bounds
    sidd_structure = create_sidd_structure(
        ortho_helper, ortho_bounds,
        product_class='Color Subaperture Image', pixel_type='RGB24I', version=version)
    # set suggested name
    sidd_structure.NITF['SUGGESTED_NAME'] = csi_calculator.sicd.get_suggested_name(csi_calculator.index)+'_CSI'

    full_filename = _validate_filename(output_directory, output_file, sidd_structure)

    if number_of_processes > 1:
        multiproc_iter = _make_multiproc_iter(ortho_iterator, number_of_processes)
        with mp.Pool(processes=number_of_processes) as pool:
            results_interators = pool.imap_unordered(_mp_worker, multiproc_iter)
            pool.close()
            pool.join()
    else:
        results_interators = [ortho_iterator]

    with SIDDWriter(full_filename, sidd_structure, csi_calculator.sicd if include_sicd else None) as writer:
        for results in results_interators:
            for data, start_indices in results:
                writer(data, start_indices=start_indices, index=0)


def create_dynamic_image_sidd(
        ortho_helper, output_directory, output_file=None, dimension=0, block_size=10,
        bounds=None, frame_count=9, aperture_fraction=0.2, method='FULL', version=3,
        include_sicd=True, remap_function=None, number_of_processes=1):
    """
    Create a SIDD version of a Dynamic Image (Sub-Aperture Stack) from a SICD type reader.

    Parameters
    ----------
    ortho_helper : OrthorectificationHelper
        The ortho-rectification helper object.
    output_directory : str
        The output directory for the given file.
    output_file : None|str
        The file name, this will default to a sensible value.
    dimension : int
        The dimension over which to split the sub-aperture.
    block_size : int
        The approximate processing block size to fetch, given in MB. The
        minimum value for use here will be 1.
    bounds : None|numpy.ndarray|list|tuple
        The sicd pixel bounds of the form `(min row, max row, min col, max col)`.
        This will default to the full image.
    frame_count : int
        The number of frames to calculate.
    aperture_fraction : float
        The relative size of each aperture window.
    method : str
        The subaperture processing method, which must be one of
        `('NORMAL', 'FULL', 'MINIMAL')`.
    version : int
        The SIDD version to use, must be one of 1, 2, or 3.
    include_sicd : bool
        Include the SICD structure in the SIDD file?
    remap_function : None|MonochromaticRemap
        The applied remap function. If one is not provided, then a default is
        used. Required global parameters will be calculated if they are missing,
        so the internal state of this remap function may be modified.
    number_of_processes: int
       The number of worker processes to use (default = 1).
       If number_of_processes = 0 then the number returned by os.cpu_count() is used.

    Returns
    -------
    None

    Examples
    --------
    Create a basic dynamic image.

    .. code-block:: python

        import os
        from sarpy.io.complex.converter import open_complex
        from sarpy.processing.sidd.sidd_product_creation import create_dynamic_image_sidd
        from sarpy.processing.sicd.csi import CSICalculator
        from sarpy.processing.ortho_rectify import NearestNeighborMethod

        reader = open_complex('<sicd type object file name>')
        ortho_helper = NearestNeighborMethod(reader, index=0)
        create_dynamic_image_sidd(ortho_helper, '<output directory>', dimension=0, version=2)
    """
    number_of_processes = _validate_number_of_processes(number_of_processes)

    if not os.path.isdir(output_directory):
        raise SarpyIOError(_output_text.format(output_directory))

    if not isinstance(ortho_helper, OrthorectificationHelper):
        raise TypeError(_orthohelper_text.format(type(ortho_helper)))

    # construct the subaperture calculator class
    subap_calculator = SubapertureCalculator(
        ortho_helper.reader, dimension=dimension, index=ortho_helper.index, block_size=block_size,
        frame_count=frame_count, aperture_fraction=aperture_fraction, method=method)

    if remap_function is None:
        remap_function = DEFAULT_DI_REMAP(override_name='DI_DEFAULT')        # pragma: no cover
    _validate_remap_function(remap_function)

    # construct the ortho-rectification iterator
    ortho_iterator = SubapertureOrthoIterator(
        ortho_helper, calculator=subap_calculator, bounds=bounds,
        remap_function=remap_function, recalc_remap_globals=False, depth_first=True)

    # create the sidd structure
    ortho_bounds = ortho_iterator.ortho_bounds
    sidd_structure = create_sidd_structure(
        ortho_helper, ortho_bounds,
        product_class='Dynamic Image', pixel_type='MONO{}I'.format(remap_function.bit_depth), version=version)
    # set suggested name
    sidd_structure.NITF['SUGGESTED_NAME'] = subap_calculator.sicd.get_suggested_name(subap_calculator.index)+'__DI'
    the_sidds = []
    for i in range(subap_calculator.frame_count):
        this_sidd = sidd_structure.copy()
        this_sidd.ProductCreation.ProductType = 'Frame {}'.format(i+1)
        the_sidds.append(this_sidd)

    # create the sidd writer
    if output_file is None:
        # noinspection PyProtectedMember
        full_filename = os.path.join(output_directory, sidd_structure.NITF['SUGGESTED_NAME']+'.nitf')
    else:
        full_filename = os.path.join(output_directory, output_file)
    if os.path.exists(os.path.expanduser(full_filename)):
        raise SarpyIOError('File {} already exists.'.format(full_filename))

    if number_of_processes > 1:
        multiproc_iter = _make_multiproc_iter(ortho_iterator, number_of_processes)
        with mp.Pool(processes=number_of_processes) as pool:
            results_interators = pool.imap_unordered(_mp_worker, multiproc_iter)
            pool.close()
            pool.join()
    else:
        results_interators = [ortho_iterator]

    with SIDDWriter(full_filename, the_sidds, subap_calculator.sicd if include_sicd else None) as writer:
        for results in results_interators:
            for data, start_indices, the_frame in results:
                writer(data, start_indices=start_indices, index=the_frame)

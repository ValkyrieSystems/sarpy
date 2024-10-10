"""Utilities for manipulating metadata"""

__classification__ = "UNCLASSIFIED"

import datetime

import sarpy
import sarpy.io.complex.sicd_elements.ImageFormation


def add_sicd_processing(sicd_meta, proc_type, *, applied=True, parameters=None):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    sicd_meta.ImageFormation.Processings = sicd_meta.ImageFormation.Processings or []
    sicd_meta.ImageFormation.Processings.append(
        sarpy.io.complex.sicd_elements.ImageFormation.ProcessingType(
            Type=f'{sarpy.__title__} {sarpy.__version__} | {proc_type} @ {now}',
            Applied=applied,
            Parameters=parameters,
        )
    )

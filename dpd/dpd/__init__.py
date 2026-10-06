"""RF power-amplifier behavioural modelling with geometric (Clifford) algebra.

Two model families over the same data and metrics:

* :mod:`dpd.models` -- the supplied MATLAB reference (Chebyshev memory
  polynomial), ported exactly and wrapped as a ``torch.nn.Module``.
* :mod:`dpd.clifford_model` -- geometric-algebra models, where the baseband
  phase equivariance of a PA is a structural property of the network rather
  than something hand-coded into an ``|x|^n x`` basis.
"""

from .data import GeoDataTB, Split, load_geodata_tb, preprocess_dov2
from .features import DEFAULT_PART_MODEL, ChebyshevLUTFeatures, PartModel
from .matlab import nmse_db
from .metrics import summarise
from .models import MemoryPolynomialPA

__all__ = [
    "GeoDataTB",
    "Split",
    "load_geodata_tb",
    "preprocess_dov2",
    "PartModel",
    "ChebyshevLUTFeatures",
    "DEFAULT_PART_MODEL",
    "MemoryPolynomialPA",
    "nmse_db",
    "summarise",
]

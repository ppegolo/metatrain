from .trainer import Trainer
from .wrapper import GLEWrapper


# On the `gle-covariant` branch the GLE architecture IS the backbone-agnostic covariant
# wrapper. The PET-bound scalar model lives on the `gle` branch; the two are separate
# lineages with different state spaces and incompatible checkpoints, deliberately.
__model__ = GLEWrapper
__trainer__ = Trainer
__capabilities__ = {
    "supported_devices": __model__.__supported_devices__,
    "supported_dtypes": __model__.__supported_dtypes__,
}

__authors__ = [
    ("Paolo Pegolo <paolo.pegolo@epfl.ch>", "@ppegolo"),
    ("Filippo Bigi <filippo.bigi@epfl.ch>", "@frostedoyster"),
    ("Sergey Pozdnyakov <sergey.pozdnyakov@epfl.ch>", "@spozdn"),
    ("Arslan Mazitov <arslan.mazitov@epfl.ch>", "@abmazitov"),
]

__maintainers__ = [
    ("Paolo Pegolo <paolo.pegolo@epfl.ch>", "@ppegolo"),
]

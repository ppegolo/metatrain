from .model import GLE
from .trainer import Trainer


__model__ = GLE
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

from enum import Enum, auto


"""" Enumerazione che contiene tuttii i sottotipi per cellule e terapie e l'istologia del tumore."""
class TCellSubtype(Enum):
    CD8 = auto()
    CD4 = auto()

class THelperSubtype(Enum):
    TH1 = auto()
    TH2 = auto()
    TREG = auto()
    TH17 = auto()

class MacrophagePhenotype(Enum):
    M1 = auto()  # Pro-infiammatorio
    M2 = auto()  # Anti-infiammatorio

class TherapyType(Enum):
    IMMUNO = auto()
    PD1_INHIBITOR = auto()
    CTLA4_INHIBITOR = auto()
    COMBO = auto()

class TumorHistology(Enum):
    CLEAR_CELL = auto()
    PAPILLARY = auto()
    CHROMOPHOBE = auto()
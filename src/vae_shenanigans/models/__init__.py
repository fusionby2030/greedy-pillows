from .base import *
from .vanilla_vae import *
from .beta_vae import *
from .conditional_vae import *


vae_models = {
    "ConditionalVAE": ConditionalVAE,
    "BetaVAE": BetaVAE,
    "VanillaVAE": VanillaVAE
}

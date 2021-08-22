from .base import *
from .vanilla_vae import *
from .beta_vae import *
from .conditional_vae import *
from .simple_ae import *

vae_models = {
    "ConditionalVAE": ConditionalVAE,
    "BetaVAE": BetaVAE,
    "VanillaVAE": VanillaVAE,
    "SimpleAE": SimpleAE
}
